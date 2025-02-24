#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "renderer.h"
#include "forward.h"
#include "backward.h"
#include "state.h"

/*
====================================================================================================
3D Renderer (approximate depth testing)
====================================================================================================
*/
uint32_t getHigherMsb(uint32_t n);
__global__ void duplicateWithKeys(
	int B, int P, int F, 
	torch::PackedTensorAccessor64<int, 2> patch_min,				// (B, 2)
	torch::PackedTensorAccessor64<int, 2> faces,					// (F, 3)
	torch::PackedTensorAccessor64<float, 3> verts_image,			// (B, P, 2)
	const float* depths, const uint32_t* offsets,					// (B * F)
	uint64_t* face_keys_unsorted, uint32_t* face_values_unsorted,
	dim3 grid
);
__global__ void identifyTileRanges(int L, uint64_t* face_list_keys, uint2* ranges);

CudaRenderer::FaceState CudaRenderer::FaceState::fromChunk(char*& chunk, size_t P)
{
	FaceState face;
	obtain(chunk, face.depths, P, 128);
	obtain(chunk, face.min_depths, P, 128);
	obtain(chunk, face.max_depths, P, 128);
	obtain(chunk, face.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, face.scan_size, face.tiles_touched, face.tiles_touched, P);
	obtain(chunk, face.scanning_space, face.scan_size, 128);
	obtain(chunk, face.face_offsets, P, 128);
	return face;
}

CudaRenderer::ImageState CudaRenderer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.final_T, N, 128);
	obtain(chunk, img.final_prev_T, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRenderer::BinningState CudaRenderer::BinningState::fromChunk(char*& chunk, size_t F)
{
	BinningState binning;
	obtain(chunk, binning.face_list, F, 128);
	obtain(chunk, binning.face_list_unsorted, F, 128);
	obtain(chunk, binning.face_list_keys, F, 128);
	obtain(chunk, binning.face_list_keys_unsorted, F, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.face_list_keys_unsorted, binning.face_list_keys,
		binning.face_list_unsorted, binning.face_list, F);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

int CudaRenderer::Renderer::forward(
	std::function<char* (size_t)> faceBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	
	const int B, int P, int F,
	const torch::Tensor& background,					// (3)
	
	// render subset of the image
	const torch::Tensor& patch_min,						// (B, 2)
	const int patch_width,
	const int patch_height,
	
	// global info (shared across all batches)
	const torch::Tensor& verts,							// (P, 3)
	const torch::Tensor& faces,							// (F, 3)
	const torch::Tensor& verts_color,					// (P, 3)
	const torch::Tensor& faces_opacity,					// (F)

	// local info (per batch)
	const torch::Tensor& verts_ndc,						// (B, P, 3)
	const torch::Tensor& verts_image,					// (B, P, 2)
	const torch::Tensor& faces_intense,					// (B, F)

	// for aa
	float aa_temperature,
	const torch::Tensor& aa_face_verts,					// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges,					// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_iszero,			// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_recip,			// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_normal,			// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_normal_c,		// (B, F, 3)

	// ray
	const torch::Tensor& image_ray_o,					// (B, H, W, 3)
	const torch::Tensor& image_ray_d,					// (B, H, W, 3)
	
	torch::Tensor& out_color,							// (B, H, W, 3)
	torch::Tensor& out_depth,							// (B, H, W)

	int len_oarea_buffer,								// K
	torch::Tensor& oarea_buffer_oarea,					// (B, H, W, K)
	torch::Tensor& oarea_buffer_tri_id,					// (B, H, W, K)
	torch::Tensor& oarea_buffer_tri_cnt,				// (B, H, W)
	torch::Tensor& oarea_buffer_doarea_dtri_verts		// (B, H, W, K, 3, 2)
)
{
	/*
	BLOCK_X: Number of pixels for a tile, horizontally
    BLOCK_Y: Number of pixels for a tile, vertically
    tile_grid: Number of tiles, horizontally and vertically
	*/
    dim3 tile_grid((patch_width + BLOCK_X - 1) / BLOCK_X, (patch_height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 batch_tile_grid((patch_width + BLOCK_X - 1) / BLOCK_X, (patch_height + BLOCK_Y - 1) / BLOCK_Y, B);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	
	/*
	Allocate auxiliary buffers for face and image states
	*/
	int BF = B * F;
	size_t chunk_size = required<FaceState>(BF);
	char* chunkptr = faceBuffer(chunk_size);
	FaceState faceState = FaceState::fromChunk(chunkptr, BF);

	int BI = B * patch_width * patch_height;
	size_t img_chunk_size = required<ImageState>(BI);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, BI);

	/*
	Preprocess face: Compute depth of each face and identify number of tiles touched
	*/
	CHECK_CUDA(FORWARD::preprocess_face(
		B, P, F,
		verts,
		verts_ndc,
		verts_image,
		faces,
		patch_min,
		tile_grid,
		faceState.depths,
		faceState.min_depths,
		faceState.max_depths,
		faceState.tiles_touched), true);
		
	// Compute prefix sum over full list of touched tile counts by faces
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(
		cub::DeviceScan::InclusiveSum(
			faceState.scanning_space, 
			faceState.scan_size, 
			faceState.tiles_touched, 
			faceState.face_offsets, 
			BF), true)

	// Retrieve total number of face instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(
		cudaMemcpy(
			&num_rendered, 
			faceState.face_offsets + BF - 1, sizeof(int), 
			cudaMemcpyDeviceToHost), true);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated face indices to be sorted
	duplicateWithKeys << <(BF + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
		B, P, F,
		patch_min.packed_accessor64<int, 2>(),
		faces.packed_accessor64<int, 2>(),
		verts_image.packed_accessor64<float, 3>(),
		faceState.depths,
		faceState.face_offsets,
		binningState.face_list_keys_unsorted,
		binningState.face_list_unsorted,
		tile_grid);
	CHECK_CUDA(, true)

	int bit = getHigherMsb(B * tile_grid.x * tile_grid.y);		// ??

	// Sort complete list of (duplicated) face indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.face_list_keys_unsorted, binningState.face_list_keys,
		binningState.face_list_unsorted, binningState.face_list,
		num_rendered, 0, 32 + bit), true)

	// Even though we use (batch * width * height) number of entries for [imgState.ranges],
	// we use only (batch * Num tile) number of entires, not entire pixels;
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, B * tile_grid.x * tile_grid.y * sizeof(uint2)), true);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
			num_rendered,
			binningState.face_list_keys,
			imgState.ranges);
	CHECK_CUDA(, true)

	// Let each tile blend its range of faces independently in parallel
	CHECK_CUDA(FORWARD::render(
		batch_tile_grid, block,
		
		B, P, F,
		background,

		patch_min,
		patch_width,
		patch_height,

		verts,
		faces,
		verts_color,
		faces_opacity,

		verts_ndc,
		verts_image,
		faces_intense,

		aa_temperature,
		aa_face_verts,
		aa_face_edges,
		aa_face_edges_iszero,
		aa_face_edges_recip,
		aa_face_edges_normal,
		aa_face_edges_normal_c,

		image_ray_o,
		image_ray_d,

		imgState.ranges,
		binningState.face_list,

		imgState.final_T,
		imgState.final_prev_T,
		imgState.n_contrib,

		out_color,
		out_depth,

		len_oarea_buffer,
		oarea_buffer_oarea,
		oarea_buffer_tri_id,
		oarea_buffer_tri_cnt,
		oarea_buffer_doarea_dtri_verts), true);

	return num_rendered;
}

void CudaRenderer::Renderer::backward(
	const int B, int P, int F, int R,
	const torch::Tensor& background,					// (3)

	// render subset of the image
	const torch::Tensor& patch_min,						// (B, 2)
	const int patch_width,
	const int patch_height,

	// global info (shared across all batches)
	const torch::Tensor& verts,							// (P, 3)
	const torch::Tensor& faces,							// (F, 3)	
	const torch::Tensor& verts_color,					// (P, 3)
	const torch::Tensor& faces_opacity,					// (F)

	// local info (per batch)
	const torch::Tensor& verts_ndc,						// (B, P, 3)
	const torch::Tensor& verts_image,					// (B, P, 2)
	const torch::Tensor& faces_intense,					// (B, F)

	// for aa
	float aa_temperature,
	const torch::Tensor& aa_face_verts,					// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges,					// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_iszero,			// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_recip,			// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_normal,			// (B, F, 3, 2)
	const torch::Tensor& aa_face_edges_normal_c,		// (B, F, 3)

	// ray
	const torch::Tensor& image_ray_o,					// (B, H, W, 3)
	const torch::Tensor& image_ray_d,					// (B, H, W, 3)

	// incoming gradient
	const torch::Tensor& dL_dout_color,					// (B, H, W, 3)
	const torch::Tensor& dL_dout_depth,					// (B, H, W)

	// buffers
	char* face_buffer,							
	char* binning_buffer,
	char* image_buffer,

	int len_oarea_buffer,									// K
	const torch::Tensor& oarea_buffer_oarea,				// (B, H, W, K)
	const torch::Tensor& oarea_buffer_tri_id,				// (B, H, W, K)
	const torch::Tensor& oarea_buffer_tri_cnt,				// (B, H, W)
	const torch::Tensor& oarea_buffer_doarea_dtri_verts,	// (B, H, W, K, 3, 2)

	torch::Tensor& dL_dverts,							// (P, 3)
	torch::Tensor& dL_dverts_color,						// (P, 3)
	torch::Tensor& dL_dfaces_opacity,					// (F)
	torch::Tensor& dL_dverts_ndc,						// (B, P, 3)
	torch::Tensor& dL_dfaces_intense,					// (B, F)
	torch::Tensor& dL_daa_face_verts					// (B, F, 3, 2)
)
{
	int BF = B * F;
	FaceState faceState = FaceState::fromChunk(face_buffer, BF);

	int BI = B * patch_width * patch_height;
	ImageState imgState = ImageState::fromChunk(image_buffer, BI);

	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	
	const dim3 batch_tile_grid((patch_width + BLOCK_X - 1) / BLOCK_X, (patch_height + BLOCK_Y - 1) / BLOCK_Y, B);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	CHECK_CUDA(BACKWARD::render(
		batch_tile_grid, block,

		B, P, F,
		background,

		patch_min,
		patch_width,
		patch_height,

		verts,
		faces,
		verts_color,
		faces_opacity,

		verts_ndc,
		verts_image,
		faces_intense,

		aa_temperature,
		aa_face_verts,
		aa_face_edges,
		aa_face_edges_iszero,
		aa_face_edges_recip,
		aa_face_edges_normal,
		aa_face_edges_normal_c,

		image_ray_o,
		image_ray_d,

		// incoming gradient
		dL_dout_color,
		dL_dout_depth,
		
		// buffers
		imgState.ranges,
		imgState.final_T,
		imgState.final_prev_T,
		imgState.n_contrib,
		binningState.face_list,

		len_oarea_buffer,
		oarea_buffer_oarea,
		oarea_buffer_tri_id,
		oarea_buffer_tri_cnt,
		oarea_buffer_doarea_dtri_verts,
		
		// outgoing gradient
		dL_dverts,
		dL_dverts_color,
		dL_dfaces_opacity,
		dL_dverts_ndc,
		dL_dfaces_intense,
		dL_daa_face_verts), true);
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all face / tile overlaps. 
// Run once per face (1:N mapping).
__global__ void duplicateWithKeys(
	int B, int P, int F, 
	torch::PackedTensorAccessor64<int, 2> patch_min,			// (B, 2)
	torch::PackedTensorAccessor64<int, 2> faces,				// (F, 3)
	torch::PackedTensorAccessor64<float, 3> verts_image,		// (B, P, 2)
	const float* depths, const uint32_t* offsets,				// (B * F)
	uint64_t* face_keys_unsorted, uint32_t* face_values_unsorted,
	dim3 grid
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= B * F)
		return;

	int batch_id = idx / F;
	int face_id = idx % F;
	uint2 batch_patch_min = make_uint2(patch_min[batch_id][0], patch_min[batch_id][1]);

	// Find this face's offset in buffer for writing keys/values.
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

	int face_vert_id_0 = faces[face_id][0];
	int face_vert_id_1 = faces[face_id][1];
	int face_vert_id_2 = faces[face_id][2];

	float2 p0 = { verts_image[batch_id][face_vert_id_0][0], verts_image[batch_id][face_vert_id_0][1] };
	float2 p1 = { verts_image[batch_id][face_vert_id_1][0], verts_image[batch_id][face_vert_id_1][1] };
	float2 p2 = { verts_image[batch_id][face_vert_id_2][0], verts_image[batch_id][face_vert_id_2][1] };
	uint2 rect_min, rect_max;
	getPatchRectFromTri(batch_patch_min, p0, p1, p2, rect_min, rect_max, grid);

	// For each tile that the bounding rect overlaps, emit a 
	// key/value pair. The key is |  tile ID  |      depth      |,
	// and the value is the ID of the face. Sorting the values 
	// with this key yields face IDs in a list, such that they
	// are first sorted by tile and then by depth. 
	int grid_size = grid.x * grid.y;
	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			uint64_t key = y * grid.x + x;
			key = key + (grid_size * batch_id);		// Add batch ID to the key
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			face_keys_unsorted[off] = key;
			face_values_unsorted[off] = face_id;
			off++;
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) face ID.
__global__ void identifyTileRanges(int L, uint64_t* face_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = face_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = face_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

/*
====================================================================================================
Render Layer Generator (precise depth testing based on tet grid) (non-differentiable)
====================================================================================================
*/
CudaRenderer::ImageRenderLayerState CudaRenderer::ImageRenderLayerState::fromChunk(char*& chunk, size_t N)
{
	ImageRenderLayerState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.first_face, N, 128);
	obtain(chunk, img.first_tet, N, 128);
	return img;
}

void CudaRenderer::RenderLayerGenerator::forward(
	std::function<char* (size_t)> faceBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	
	const int B, int P, int F, int T,
	const int width, int height,

	const torch::Tensor& verts,							// (P, 3)
	const torch::Tensor& faces,							// (F, 3)
	const torch::Tensor& tets,							// (T, 4)
	const torch::Tensor& face_tets,						// (F, 2)
	const torch::Tensor& tet_faces,						// (T, 4)
	const torch::Tensor& faces_existence,				// (F,)

	// local info (per batch)
	const torch::Tensor& verts_ndc,						// (B, P, 3)
	const torch::Tensor& verts_image,					// (B, P, 2)

	// ray
	const torch::Tensor& image_ray_o,					// (B, H, W, 3)
	const torch::Tensor& image_ray_d,					// (B, H, W, 3)
	
	int num_layers,										// L
	torch::Tensor& render_layers,						// (B, H, W, L)
	torch::Tensor& render_layers_cnt					// (B, H, W)
) {
	int BF = B * F;
	size_t face_chunk_size = required<FaceState>(BF);
	char* face_chunkptr = faceBuffer(face_chunk_size);
	FaceState faceState = FaceState::fromChunk(face_chunkptr, BF);

	int BI = B * width * height;
	size_t img_chunk_size = required<ImageRenderLayerState>(BI);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageRenderLayerState imgState = ImageRenderLayerState::fromChunk(img_chunkptr, BI);

	// ===================
	// Find starting face of each ray
	// ===================

    // BLOCK_X: Number of pixels for a tile, horizontally
    // BLOCK_Y: Number of pixels for a tile, vertically
    // tile_grid: Number of tiles, horizontally and vertically
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 batch_tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, B);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	auto int_options = verts.options().dtype(torch::kInt32);
	torch::Tensor patch_min = torch::zeros({ B, 2 }, int_options);

	CHECK_CUDA(FORWARD::preprocess_face(
		B, P, F,
		verts,
		verts_ndc,
		verts_image,
		faces,
		patch_min,
		tile_grid,
		faceState.depths,
		faceState.min_depths,
		faceState.max_depths,
		faceState.tiles_touched), true);
	
	// Compute prefix sum over full list of 
	// touched tile counts by faces into face_offsets.
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(
		cub::DeviceScan::InclusiveSum(
			faceState.scanning_space, 
			faceState.scan_size, 
			faceState.tiles_touched, 
			faceState.face_offsets, 
			B * F), true)

	// Retrieve total number of face instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(
		cudaMemcpy(
			&num_rendered, 
			faceState.face_offsets + (B * F) - 1, sizeof(int), 
			cudaMemcpyDeviceToHost), true);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated face indices to be sorted
	duplicateWithKeys << <(BF + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
		B, P, F,
		patch_min.packed_accessor64<int, 2>(),
		faces.packed_accessor64<int, 2>(),
		verts_image.packed_accessor64<float, 3>(),
		faceState.min_depths,			// sort by min depth
		faceState.face_offsets,
		binningState.face_list_keys_unsorted,
		binningState.face_list_unsorted,
		tile_grid);
	CHECK_CUDA(, true)

	int bit = getHigherMsb(B * tile_grid.x * tile_grid.y);		// ??

	// Sort complete list of (duplicated) face indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.face_list_keys_unsorted, binningState.face_list_keys,
		binningState.face_list_unsorted, binningState.face_list,
		num_rendered, 0, 32 + bit), true)

	// Even though we use (batch * width * height) number of entries for [imgState.ranges],
	// we use only (batch * Num tile) number of entries, not entire pixels;
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, B * tile_grid.x * tile_grid.y * sizeof(uint2)), true);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.face_list_keys,
			imgState.ranges);
	CHECK_CUDA(, true)

	// Identify first triangle face that each ray collides with
	CHECK_CUDA(FORWARD::first_intersect(
		batch_tile_grid, block,
		verts,
		faces,
		tets,
		face_tets,
		faceState.min_depths,
		faceState.max_depths,
		imgState.ranges,
		binningState.face_list,
		B, F, width, height,
		image_ray_o,
		image_ray_d,
		imgState.first_face,
		imgState.first_tet), true);
		
	// ===================
	// Forward rendering by ray marching
	// ===================
	CHECK_CUDA(FORWARD::generate_render_layers(
		batch_tile_grid, block,
		
		verts,
		faces,
		tets,
		face_tets,
		tet_faces,
		faces_existence,

		image_ray_o,
		image_ray_d,

		imgState.first_face,
		imgState.first_tet,

		width, height,
		P, F,

		num_layers,
		render_layers,
		render_layers_cnt), true);
}