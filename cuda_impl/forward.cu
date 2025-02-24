#include "forward.h"
#include "auxiliary.h"
#include "cuda_math.h"
#include "aa.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <tuple>

namespace FORWARD {
    /*
    ========================================================
    3D Rendering
    ========================================================
    */
    __global__ void preprocessFaceCUDA(
		int B, int P, int F,
        torch::PackedTensorAccessor64<float, 2> verts,              // (P, 3)
        torch::PackedTensorAccessor64<float, 3> verts_ndc,          // (B, P, 3)
        torch::PackedTensorAccessor64<float, 3> verts_image,        // (B, P, 2)
        torch::PackedTensorAccessor64<int, 2> faces,                // (F, 3)
        torch::PackedTensorAccessor64<int, 2> patch_min,            // (B, 2)
        const dim3 grid,
        float* depths,
		float* min_depths,
		float* max_depths,
        uint32_t* tiles_touched)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= B * F)
			return;

		int batch_id = idx / F;
		int face_id = idx % F;
        uint2 batch_patch_min = { patch_min[batch_id][0], patch_min[batch_id][1] };

		// Initialize touched tiles to 0. If this isn't changed,
		// this triangle will not be processed further.
		tiles_touched[idx] = 0;

		// Get point info;
		float max_z = 0;
		float min_z = 0;
		float depth = 0;
		float2 face_verts_image[3];
		for (int i = 0; i < 3; i++) {
            int face_vert_id = faces[face_id][i];
            float3 face_vert_ndc = {
                verts_ndc[batch_id][face_vert_id][0],
                verts_ndc[batch_id][face_vert_id][1],
                verts_ndc[batch_id][face_vert_id][2]
            };
			float z = face_vert_ndc.z;
			if (i == 0) {
				max_z = z;
				min_z = z;
			}
			else {
				max_z = max(max_z, z);
				min_z = min(min_z, z);
			}
			depth += z;
			face_verts_image[i] = {
                verts_image[batch_id][face_vert_id][0],
                verts_image[batch_id][face_vert_id][1]
            };
		}
		depth = depth / 3.0f;

		// If triangle is completely behind or front of camera, quit.
		if (max_z < -1.0f || min_z > 1.0f)
			return;

		// Compute a bounding rectangle of screen-space tiles that this 
		// triangle overlaps with. Quit if rectangle covers 0 tiles.
		uint2 rect_min, rect_max;
        getPatchRectFromTri(
            batch_patch_min,
            face_verts_image[0],
            face_verts_image[1],
            face_verts_image[2],
            rect_min,
            rect_max,
            grid
        );

		// If triangle is completely outside of camera, quit.
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			return;

		// Store some useful helper data for the next steps.
		depths[idx] = depth;
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

		// change range of [depth] from [-1, 1] to [0, 1] for ordering later;
		depths[idx] = (depth + 1.0f) * 0.5f;
		if (depths[idx] < 0.0f) depths[idx] = 0.0f;
		if (depths[idx] > 1.0f) depths[idx] = 1.0f;

		// min & max depths
		min_depths[idx] = (min_z + 1.0f) * 0.5f;
		if (min_depths[idx] < 0.0f) min_depths[idx] = 0.0f;
		if (min_depths[idx] > 1.0f) min_depths[idx] = 1.0f;

		max_depths[idx] = (max_z + 1.0f) * 0.5f;
		if (max_depths[idx] < 0.0f) max_depths[idx] = 0.0f;
		if (max_depths[idx] > 1.0f) max_depths[idx] = 1.0f;
	}

    void preprocess_face(
		int B, int P, int F,
		const torch::Tensor& verts,             // (P, 3)
		const torch::Tensor& verts_ndc,         // (B, P, 3)
		const torch::Tensor& verts_image,       // (B, P, 2)
		const torch::Tensor& faces,             // (F, 3)
		const torch::Tensor& patch_min,         // (B, 2)
		const dim3 grid,
		float* depths,
		float* min_depths,
		float* max_depths,
		uint32_t* tiles_touched
	) {
        int BF = B * F;
		preprocessFaceCUDA << <(BF + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
			B, P, F,
			verts.packed_accessor64<float, 2>(),
			verts_ndc.packed_accessor64<float, 3>(),
			verts_image.packed_accessor64<float, 3>(),
			faces.packed_accessor64<int, 2>(),
			patch_min.packed_accessor64<int, 2>(),
			grid,
			depths,
			min_depths,
			max_depths,
			tiles_touched
        );
    }

    template <uint32_t CHANNELS>
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	renderCUDA(
        int B, int P, int F,
        torch::PackedTensorAccessor64<float, 1> background,				    // (3)

        torch::PackedTensorAccessor64<int, 2> patch_min,					// (B, 2)
        const int patch_width,
        const int patch_height,

        torch::PackedTensorAccessor64<float, 2> verts,						// (P, 3)
        torch::PackedTensorAccessor64<int, 2> faces,						// (F, 3)
        torch::PackedTensorAccessor64<float, 2> verts_color,				// (P, 3)
        torch::PackedTensorAccessor64<float, 1> faces_opacity,				// (F,)

        torch::PackedTensorAccessor64<float, 3> verts_ndc, 			  	    // (B, P, 3)
        torch::PackedTensorAccessor64<float, 3> verts_image,				// (B, P, 2)
        torch::PackedTensorAccessor64<float, 2> faces_intense,				// (B, F,)

		float aa_temperature,
        torch::PackedTensorAccessor64<float, 4> aa_face_verts,				// (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 4> aa_face_edges,				// (B, F, 3, 2)
        torch::PackedTensorAccessor64<bool, 4> aa_face_edges_iszero,		// (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 4> aa_face_edges_recip,		// (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 4> aa_face_edges_normal,		// (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 3> aa_face_edges_normal_c,	    // (B, F, 3)

        torch::PackedTensorAccessor64<float, 3> aa_face_verts_min,          // (B, F, 2)
        torch::PackedTensorAccessor64<float, 3> aa_face_verts_max,          // (B, F, 2)

        torch::PackedTensorAccessor64<float, 4> ray_o, 					    // (B, H, W, 3)
        torch::PackedTensorAccessor64<float, 4> ray_d, 					    // (B, H, W, 3)

        const uint2* ranges,
        const uint32_t* face_list,

        float* final_T,
        float* final_prev_T,
        uint32_t* n_contrib,

        torch::PackedTensorAccessor64<float, 4> out_color,					// (B, H, W, 3)
        torch::PackedTensorAccessor64<float, 3> out_depth,					// (B, H, W)

		int len_oarea_buffer,														// K
		torch::PackedTensorAccessor64<float, 4> oarea_buffer_oarea,					// (B, H, W, K)
		torch::PackedTensorAccessor64<int, 4> oarea_buffer_tri_id,					// (B, H, W, K)
		torch::PackedTensorAccessor64<int, 3> oarea_buffer_tri_cnt,					// (B, H, W)
		torch::PackedTensorAccessor64<float, 6> oarea_buffer_doarea_dtri_verts		// (B, H, W, K, 3, 2)
    )
	{
		// Identify current tile and associated min/max pixel range.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;

        uint2 b_patch_min = { patch_min[batch_id][0], patch_min[batch_id][1] };

		uint32_t horizontal_blocks = (patch_width + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (patch_height + BLOCK_Y - 1) / BLOCK_Y;

		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		
		uint32_t pix_id = (patch_width * pix.y) + pix.x;
		uint32_t b_pix_id = (batch_id * patch_width * patch_height) + pix_id;

		// Check if this thread is associated with a valid pixel or outside.
		bool inside = (batch_id < B) && (pix.x < patch_width) && (pix.y < patch_height);
		// Done threads can help with fetching, but don't rasterize
		bool done = !inside;

		float3 this_ray_o, this_ray_d;
		if (inside) {
            this_ray_o.x = ray_o[batch_id][pix.y][pix.x][0];
            this_ray_o.y = ray_o[batch_id][pix.y][pix.x][1];
            this_ray_o.z = ray_o[batch_id][pix.y][pix.x][2];

            this_ray_d.x = ray_d[batch_id][pix.y][pix.x][0];
            this_ray_d.y = ray_d[batch_id][pix.y][pix.x][1];
            this_ray_d.z = ray_d[batch_id][pix.y][pix.x][2];
		}

		// Load start/end range of IDs to process in bit sorted list.
		int b_tile_id = (batch_id * horizontal_blocks * vertical_blocks) + 
							(block.group_index().y * horizontal_blocks) + block.group_index().x;
		uint2 range = ranges[b_tile_id];
		const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
		int toDo = range.y - range.x;

		// Allocate storage for batches of collectively fetched data.
		__shared__ int collected_face_id[BLOCK_SIZE];

		__shared__ float3 collected_face_vert_0[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_1[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_2[BLOCK_SIZE];

		__shared__ float collected_face_vert_color_0[BLOCK_SIZE * CHANNELS];
		__shared__ float collected_face_vert_color_1[BLOCK_SIZE * CHANNELS];
		__shared__ float collected_face_vert_color_2[BLOCK_SIZE * CHANNELS];

		__shared__ float collected_face_vert_depth_0[BLOCK_SIZE];
		__shared__ float collected_face_vert_depth_1[BLOCK_SIZE];
		__shared__ float collected_face_vert_depth_2[BLOCK_SIZE];
		
		__shared__ float collected_face_opacity[BLOCK_SIZE];
		__shared__ float collected_face_intense[BLOCK_SIZE];

		// Initialize helper variables
		float pT = 1.0f;
		float T = 1.0f;
		uint32_t contributor = 0;
		uint32_t last_contributor = 0;
		float C[CHANNELS] = { 0 };
		float D[1] = { 0 };
		int t_grad_oarea_tri_cnt = 0;

		// Iterate over batches until all done or range is complete
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// End if entire block votes that it is done rasterizing
			int num_done = __syncthreads_count(done);
			if (num_done == BLOCK_SIZE)
				break;

			// Collectively fetch per-face data from global to shared
			int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				int face_id = face_list[range.x + progress];

				collected_face_id[block.thread_rank()] = face_id;

				int face_vert_id_0 = faces[face_id][0];
                int face_vert_id_1 = faces[face_id][1];
                int face_vert_id_2 = faces[face_id][2];

                // Fetch face vertices
                collected_face_vert_0[block.thread_rank()].x = verts[face_vert_id_0][0];
                collected_face_vert_0[block.thread_rank()].y = verts[face_vert_id_0][1];
                collected_face_vert_0[block.thread_rank()].z = verts[face_vert_id_0][2];

                collected_face_vert_1[block.thread_rank()].x = verts[face_vert_id_1][0];
                collected_face_vert_1[block.thread_rank()].y = verts[face_vert_id_1][1];
                collected_face_vert_1[block.thread_rank()].z = verts[face_vert_id_1][2];

                collected_face_vert_2[block.thread_rank()].x = verts[face_vert_id_2][0];
                collected_face_vert_2[block.thread_rank()].y = verts[face_vert_id_2][1];
                collected_face_vert_2[block.thread_rank()].z = verts[face_vert_id_2][2];

                // Fetch face vertices' color
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					collected_face_vert_color_0[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_0][ch];
                    collected_face_vert_color_1[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_1][ch];
                    collected_face_vert_color_2[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_2][ch];
				}

                // Fetch face vertices' depth
				collected_face_vert_depth_0[block.thread_rank()] = verts_ndc[batch_id][face_vert_id_0][2];
                collected_face_vert_depth_1[block.thread_rank()] = verts_ndc[batch_id][face_vert_id_1][2];
                collected_face_vert_depth_2[block.thread_rank()] = verts_ndc[batch_id][face_vert_id_2][2];

                // Fetch face opacity and intensity
				collected_face_opacity[block.thread_rank()] = faces_opacity[face_id];
				collected_face_intense[block.thread_rank()] = faces_intense[batch_id][face_id];
			}
			block.sync();

			// Iterate over current batch
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				// Keep track of current position in range
				contributor++;

                int face_id = collected_face_id[j];

                float txmin = aa_face_verts_min[batch_id][face_id][0];
                float txmax = aa_face_verts_max[batch_id][face_id][0];
                float tymin = aa_face_verts_min[batch_id][face_id][1];
                float tymax = aa_face_verts_max[batch_id][face_id][1];

                float pxmin = pix.x + b_patch_min.x;
                float pxmax = pxmin + 1;
                float pymin = pix.y + b_patch_min.y;
                float pymax = pymin + 1;
                float pix_area = 1.0f;

                // Compute overlapping area between the current face and the current pixel
                float oarea = 0;
                float t_grad_oarea_tri_verts[3][2] = { { 0, 0 }, { 0, 0 }, { 0, 0 } };
				int error_code = 0;
				if (aa_temperature > 0.0f) {
					error_code = AA::tri_pix_overlap_area(
						aa_face_verts, aa_face_edges, aa_face_edges_iszero, aa_face_edges_recip, aa_face_edges_normal, aa_face_edges_normal_c,
						batch_id, face_id,
						txmin, txmax, tymin, tymax,
						pxmin, pxmax, pymin, pymax, pix_area,
						&oarea, t_grad_oarea_tri_verts
					);
					if ((error_code != 0) || (oarea == 0.0f))
	                    continue;
				}
				
				float oarea_ratio = oarea / pix_area;
                
				// save gradient for backward pass
				if ((aa_temperature > 0.0f) && (t_grad_oarea_tri_cnt < len_oarea_buffer))
				{
					oarea_buffer_oarea[batch_id][pix.y][pix.x][t_grad_oarea_tri_cnt] = oarea;
					oarea_buffer_tri_id[batch_id][pix.y][pix.x][t_grad_oarea_tri_cnt] = face_id;
					for (int ii = 0; ii < 3; ii++)
						for (int jj = 0; jj < 2; jj++)
							oarea_buffer_doarea_dtri_verts[batch_id][pix.y][pix.x][t_grad_oarea_tri_cnt][ii][jj] = t_grad_oarea_tri_verts[ii][jj];
					t_grad_oarea_tri_cnt++;
				}

				// Find intersection point between ray and triangle;
				// It is used to interpolate vert-wise color and depth;
				const float3& v0 = collected_face_vert_0[j];
				const float3& v1 = collected_face_vert_1[j];
				const float3& v2 = collected_face_vert_2[j];
				float3 tuv = { 0, 0, 0 };
				bool not_edge_case = ray_tri_intersection(
					this_ray_o, this_ray_d,
					v0, v1, v2, tuv);

				if (!not_edge_case)
					continue;

				// clamp, because (iu, iv) could be outside of the triangle.
				float iu = tuv.y, iv = tuv.z;
				float iuc, ivc;
				int iclamp_code;
				clamp_bary_uv(iu, iv, iuc, ivc, iclamp_code);	
				float i0 = 1 - iuc - ivc, i1 = iuc, i2 = ivc;

                // if iclamp_code == 0, then the intersection point is inside the triangle
				if (iclamp_code == 0)
					oarea_ratio = 1.0 * (1.0f - aa_temperature) + oarea_ratio * aa_temperature;
				else
					oarea_ratio = 0.0 * (1.0f - aa_temperature) + oarea_ratio * aa_temperature;

				if (oarea_ratio == 0.0f)
					continue;
				
				// Find intersection point's color, depth, and normal;
				float iC[CHANNELS] = { 0 };
				float iD[1] = { 0 };
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					iC[ch] = i0 * collected_face_vert_color_0[j * CHANNELS + ch] +
							i1 * collected_face_vert_color_1[j * CHANNELS + ch] +
							i2 * collected_face_vert_color_2[j * CHANNELS + ch];
					iC[ch] = iC[ch] * collected_face_intense[j];
				}
				iD[0] = i0 * collected_face_vert_depth_0[j] +
						i1 * collected_face_vert_depth_1[j] +
						i2 * collected_face_vert_depth_2[j];
						
				// Find intersecting face's opacity;
				float alpha = collected_face_opacity[j] * oarea_ratio;
				float test_T = T * (1 - alpha);

				// alpha blending.
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += iC[ch] * alpha * T;
				D[0] += iD[0] * alpha * T;
				
				pT = T;
				T = test_T;

				// Keep track of last range entry to update this pixel.
				last_contributor = contributor;

				if (T < T_EPS) {
					done = true;
					break;
				}
			}
		}

		// Write out results to global memory
		if (inside)
		{
			final_prev_T[b_pix_id] = pT;
			final_T[b_pix_id] = T;
			n_contrib[b_pix_id] = last_contributor;
			
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[batch_id][pix.y][pix.x][ch] = C[ch] + T * background[ch];
			out_depth[batch_id][pix.y][pix.x] = D[0] + T * 1.0f;

			oarea_buffer_tri_cnt[batch_id][pix.y][pix.x] = t_grad_oarea_tri_cnt;
		}
	}

    void render(
		const dim3 grid, dim3 block,

		const int B, int P, int F,
		const torch::Tensor& background,				// (3)

		const torch::Tensor& patch_min,					// (B, 2)
		const int patch_width,
		const int patch_height,
		
		const torch::Tensor& verts,						// (P, 3)
		const torch::Tensor& faces,						// (F, 3)
		const torch::Tensor& verts_color,				// (P, 3)
		const torch::Tensor& faces_opacity,				// (F,)
		
		const torch::Tensor& verts_ndc, 			  	// (B, P, 3)
		const torch::Tensor& verts_image,				// (B, P, 2)
		const torch::Tensor& faces_intense,				// (B, F,)

		float aa_temperature,
		const torch::Tensor& aa_face_verts,				// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges,				// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_iszero,		// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_recip,		// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_normal,		// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_normal_c,	// (B, F, 3)

		const torch::Tensor& ray_o, 					// (B, H, W, 3)
		const torch::Tensor& ray_d, 					// (B, H, W, 3)

		const uint2* ranges,
		const uint32_t* face_list,
		
		float* final_T,
		float* final_prev_T,
		uint32_t* n_contrib,
		
		torch::Tensor& out_color,						// (B, H, W, 3)
		torch::Tensor& out_depth,						// (B, H, W)

		int len_oarea_buffer,							// K
		torch::Tensor& oarea_buffer_oarea,				// (B, H, W, K)
		torch::Tensor& oarea_buffer_tri_id,				// (B, H, W, K)
		torch::Tensor& oarea_buffer_tri_cnt,			// (B, H, W)
		torch::Tensor& oarea_buffer_doarea_dtri_verts	// (B, H, W, K, 3, 2)
	) {
		torch::Tensor aa_face_verts_min = std::get<0>(aa_face_verts.min(2));            // (B, F, 2)
		torch::Tensor aa_face_verts_max = std::get<0>(aa_face_verts.max(2));            // (B, F, 2)

        renderCUDA<3> << <grid, block >> > (
            B, P, F,
            background.packed_accessor64<float, 1>(),

            patch_min.packed_accessor64<int, 2>(),
            patch_width,
            patch_height,

            verts.packed_accessor64<float, 2>(),
            faces.packed_accessor64<int, 2>(),
            verts_color.packed_accessor64<float, 2>(),
            faces_opacity.packed_accessor64<float, 1>(),

            verts_ndc.packed_accessor64<float, 3>(),
            verts_image.packed_accessor64<float, 3>(),
            faces_intense.packed_accessor64<float, 2>(),

			aa_temperature,
            aa_face_verts.packed_accessor64<float, 4>(),
            aa_face_edges.packed_accessor64<float, 4>(),
            aa_face_edges_iszero.packed_accessor64<bool, 4>(),
            aa_face_edges_recip.packed_accessor64<float, 4>(),
            aa_face_edges_normal.packed_accessor64<float, 4>(),
            aa_face_edges_normal_c.packed_accessor64<float, 3>(),

			aa_face_verts_min.packed_accessor64<float, 3>(),
			aa_face_verts_max.packed_accessor64<float, 3>(),

            ray_o.packed_accessor64<float, 4>(),
            ray_d.packed_accessor64<float, 4>(),

            ranges,
            face_list,

            final_T,
            final_prev_T,
            n_contrib,

            out_color.packed_accessor64<float, 4>(),
            out_depth.packed_accessor64<float, 3>(),

			len_oarea_buffer,
			oarea_buffer_oarea.packed_accessor64<float, 4>(),
			oarea_buffer_tri_id.packed_accessor64<int, 4>(),
			oarea_buffer_tri_cnt.packed_accessor64<int, 3>(),
			oarea_buffer_doarea_dtri_verts.packed_accessor64<float, 6>()
		);
	}

    /*
    ==============================================
    Render Layer Generator (precise depth testing based on tet grid) (non-differentiable)
    ==============================================
    */
    // Find first intersection info.
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	firstIntersectCUDA(
		torch::PackedTensorAccessor64<float, 2> verts,	
		torch::PackedTensorAccessor64<int, 2> faces,
		torch::PackedTensorAccessor64<int, 2> tets,
		torch::PackedTensorAccessor64<int, 2> face_tets,
		const float* faces_min_depth,
		const float* faces_max_depth,
		const uint2* ranges,
		const uint32_t* face_list,
		int B, int F, int W, int H,
		torch::PackedTensorAccessor64<float, 4> ray_o,
		torch::PackedTensorAccessor64<float, 4> ray_d,
		int* first_face,
		int* first_tet
	)			
	{
		// Identify current tile and associated min/max pixel range.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;
		
		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;
		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };

		uint32_t pix_id = W * pix.y + pix.x;
		auto pix_batch_id = (batch_id * W * H) + pix_id;

		// Check if this thread is associated with a valid pixel or outside.
		bool inside = pix.x < W && pix.y < H;
		// Done threads can help with fetching, but don't rasterize
		bool done = !inside;
		
		float3 this_ray_o, this_ray_d;
		if (inside) {
            this_ray_o.x = ray_o[batch_id][pix.y][pix.x][0];
            this_ray_o.y = ray_o[batch_id][pix.y][pix.x][1];
            this_ray_o.z = ray_o[batch_id][pix.y][pix.x][2];

            this_ray_d.x = ray_d[batch_id][pix.y][pix.x][0];
            this_ray_d.y = ray_d[batch_id][pix.y][pix.x][1];
            this_ray_d.z = ray_d[batch_id][pix.y][pix.x][2];
		}
		
		first_face[pix_batch_id] = -1;
		first_tet[pix_batch_id] = -1;		// If there is no corresponding info, remain -1;
		
		// Load start/end range of IDs to process in bit sorted list.
		auto tile_id = (batch_id * horizontal_blocks * vertical_blocks) + 
						(block.group_index().y * horizontal_blocks + block.group_index().x);
		uint2 range = ranges[tile_id];
		const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
		int toDo = range.y - range.x;

		// Allocate storage for batches of collectively fetched data.
		__shared__ int collected_face_id[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_0[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_1[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_2[BLOCK_SIZE];
		__shared__ float collected_face_min_depth[BLOCK_SIZE];
		__shared__ float collected_face_max_depth[BLOCK_SIZE];

		// Initialize helper variables
		float min_T = -1.0f;
		float min_T_max_depth = -1.0f;		// max depth of the face that has the min_T;

		// Iterate over batches until all done or range is complete
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// End if entire block votes that it is done rasterizing
			int num_done = __syncthreads_count(done);
			if (num_done == BLOCK_SIZE)
				break;

			// Collectively fetch per-face data from global to shared
			int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				int face_id = face_list[range.x + progress];
				int face_batch_id = batch_id * F + face_id;

				collected_face_id[block.thread_rank()] = face_id;

				int face_vert_id_0 = faces[face_id][0];
                int face_vert_id_1 = faces[face_id][1];
                int face_vert_id_2 = faces[face_id][2];

                // Fetch face vertices
                collected_face_vert_0[block.thread_rank()].x = verts[face_vert_id_0][0];
                collected_face_vert_0[block.thread_rank()].y = verts[face_vert_id_0][1];
                collected_face_vert_0[block.thread_rank()].z = verts[face_vert_id_0][2];

                collected_face_vert_1[block.thread_rank()].x = verts[face_vert_id_1][0];
                collected_face_vert_1[block.thread_rank()].y = verts[face_vert_id_1][1];
                collected_face_vert_1[block.thread_rank()].z = verts[face_vert_id_1][2];

                collected_face_vert_2[block.thread_rank()].x = verts[face_vert_id_2][0];
                collected_face_vert_2[block.thread_rank()].y = verts[face_vert_id_2][1];
                collected_face_vert_2[block.thread_rank()].z = verts[face_vert_id_2][2];

				collected_face_min_depth[block.thread_rank()] = faces_min_depth[face_batch_id];
				collected_face_max_depth[block.thread_rank()] = faces_max_depth[face_batch_id];
			}
			block.sync();

			// Iterate over current batch
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				if (min_T >= 0.0f && collected_face_min_depth[j] > min_T_max_depth) {
					done = true;
					continue;
				}

				// Compute barycentric coordinates and ray parameter (t) at intersection point;
				float3 tuv;
				bool not_edge_case = ray_tri_intersection(
					this_ray_o,
					this_ray_d,
					collected_face_vert_0[j],
					collected_face_vert_1[j],
					collected_face_vert_2[j],
					tuv
				);

				if (!not_edge_case)
					continue;

				bool intersect = (tuv.x >= 0.0f && tuv.y >= 0.0f && tuv.z >= 0.0f && tuv.y + tuv.z <= 1.0f);

				// If no intersection, skip this face;
				if (!intersect)
					continue;

				float curr_T = tuv.x;
				if (min_T < 0.0f || curr_T < min_T) {
					min_T = curr_T;
					min_T_max_depth = collected_face_max_depth[j];

					first_face[pix_batch_id] = collected_face_id[j];
				}
			}
		}

		// Identify the first tet that contains the first face;
		int this_first_face = first_face[pix_batch_id];

		if (this_first_face < 0)
			return;

		for (int i = 0; i < 2; i++) {
			int tet_id = face_tets[this_first_face][i];
			if (tet_id < 0)
				continue;
			
			float3 my_first_face_outward_normal;
			tet_face_outward_normal(
				verts,
				faces,
				tets,
				this_first_face,
				tet_id,
				my_first_face_outward_normal
			);

			float dot_prod = dot(my_first_face_outward_normal, this_ray_d);
			
			if (dot_prod < 0.0f)
				first_tet[pix_batch_id] = tet_id;
		}
	}

	void first_intersect(
		const dim3 grid, dim3 block,
		const torch::Tensor& verts,					// (P, 3)
		const torch::Tensor& faces,					// (F, 3)
		const torch::Tensor& tets,					// (T, 4)
		const torch::Tensor& face_tets,				// (F, 2)
		const float* faces_min_depth,				
		const float* faces_max_depth,				
		const uint2* ranges,						
		const uint32_t* face_list,
		int B, int F, int W, int H,
		const torch::Tensor& ray_o,					// (B, H, W, 3)
		const torch::Tensor& ray_d,					// (B, H, W, 3)
		int* first_face,
		int* first_tet
	) {
		firstIntersectCUDA << <grid, block >> > (
			verts.packed_accessor64<float, 2>(),
			faces.packed_accessor64<int, 2>(),
			tets.packed_accessor64<int, 2>(),
			face_tets.packed_accessor64<int, 2>(),
			faces_min_depth,
			faces_max_depth,
			ranges,
			face_list,
			B, F, W, H,
			ray_o.packed_accessor64<float, 4>(),
			ray_d.packed_accessor64<float, 4>(),
			first_face,
			first_tet
		);
	}

	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	generateRenderLayersCUDA(
		torch::PackedTensorAccessor64<float, 2> verts,
		torch::PackedTensorAccessor64<int, 2> faces,
		torch::PackedTensorAccessor64<int, 2> tets,
		torch::PackedTensorAccessor64<int, 2> face_tets,
		torch::PackedTensorAccessor64<int, 2> tet_faces,
		torch::PackedTensorAccessor64<int, 1> faces_existence,

		torch::PackedTensorAccessor64<float, 4> ray_o,
		torch::PackedTensorAccessor64<float, 4> ray_d,

		const int* __restrict__ first_face,
		const int* __restrict__ first_tet,

		int W, int H,
		int P, int F,

		const int num_layers,
		torch::PackedTensorAccessor64<int, 4> render_layers,
		torch::PackedTensorAccessor64<int, 3> render_layers_cnt)
	{
		// Identify current tile and associated min/max pixel range.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;

		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;

		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		uint32_t pix_id = W * pix.y + pix.x;
		uint32_t pix_batch_id = (batch_id * W * H) + pix_id;
		
		// Check if this thread is associated with a valid pixel or outside.
		bool inside = pix.x < W && pix.y < H;

		// If outside, do nothing;
		if (!inside)
			return;
		bool done = false;

		// ================
		// Ray marching
		// ================
		float3 this_ray_o, this_ray_d;
		if (inside) {
			this_ray_o.x = ray_o[batch_id][pix.y][pix.x][0];
			this_ray_o.y = ray_o[batch_id][pix.y][pix.x][1];
			this_ray_o.z = ray_o[batch_id][pix.y][pix.x][2];

			this_ray_d.x = ray_d[batch_id][pix.y][pix.x][0];
			this_ray_d.y = ray_d[batch_id][pix.y][pix.x][1];
			this_ray_d.z = ray_d[batch_id][pix.y][pix.x][2];
		}

		int this_first_face = first_face[pix_batch_id];
		int this_first_tet = first_tet[pix_batch_id];

		float rt = 0.0f;
		float iu, iv;
		if (this_first_face == -1 || this_first_tet == -1)
			done = true;
		else {
			// compute [rt], the ray param at the first intersection point;
			float3 tuv, p0, p1, p2;
			
			int face_vert_id0 = faces[this_first_face][0];
			int face_vert_id1 = faces[this_first_face][1];
			int face_vert_id2 = faces[this_first_face][2];
			
			p0.x = verts[face_vert_id0][0];
			p0.y = verts[face_vert_id0][1];
			p0.z = verts[face_vert_id0][2];

			p1.x = verts[face_vert_id1][0];
			p1.y = verts[face_vert_id1][1];
			p1.z = verts[face_vert_id1][2];

			p2.x = verts[face_vert_id2][0];
			p2.y = verts[face_vert_id2][1];
			p2.z = verts[face_vert_id2][2];

			ray_tri_intersection(
				this_ray_o,
				this_ray_d,
				p0,
				p1, 
				p2,
				tuv
			);
			rt = tuv.x;
			iu = tuv.y;
			iv = tuv.z;
		}

		// current tet info;
		int curr_face = this_first_face;
		int curr_tet = this_first_tet;
		float curr_rt = rt;
		float curr_iu = iu;
		float curr_iv = iv;

		int num_layers_done = 0;
		while(!done) {
			/*
			1. If current face exists, record it
			*/ 
			int curr_face_exists = faces_existence[curr_face];

			if (curr_face_exists) {
				render_layers[batch_id][pix.y][pix.x][num_layers_done] = curr_face;
				num_layers_done++;
				if (num_layers_done >= num_layers)
					done = true;
			}

			/*
			2. Find next face
			*/
			
			// Compute intersection point between the ray and
			// the remaining 3 faces in the current tet;
			int next_face = -1;
			int next_tet = -1;
			float next_rt, next_iu, next_iv;

			// If there is no current tet to explore, we are done;
			if (curr_tet == -1) {
				done = true;
			}
			
			if (!done) {
				int curr_tet_faces[4];
				curr_tet_faces[0] = tet_faces[curr_tet][0];
				curr_tet_faces[1] = tet_faces[curr_tet][1];
				curr_tet_faces[2] = tet_faces[curr_tet][2];
				curr_tet_faces[3] = tet_faces[curr_tet][3];

				int curr_tet_other_faces[3];
				int cnt = 0;
				for (int i = 0; i < 4; i++) {
					if (curr_tet_faces[i] == curr_face)
						continue;
					curr_tet_other_faces[cnt++] = curr_tet_faces[i];
				}

				if (cnt != 3) {
					// it should not happen, but we can't believe numerics...
					done = true;
					// printf("Error case 1\n");
				}

				/*
				Among three other faces in the current tet,
				find the one that intersects with the ray,
				and its outward normal is in the same direction
				as the ray direction;
				We do not use [rt] here, because it is weak to
				numerical errors;
				*/

				// if curr face's outward normal was in the same 
				// direction as the ray, error;
				float3 curr_face_outward_normal;
				tet_face_outward_normal(
					verts,
					faces,
					tets,
					curr_face,
					curr_tet,
					curr_face_outward_normal
				);
				float curr_face_normal_dot_prod = dot(curr_face_outward_normal, this_ray_d);
				if (curr_face_normal_dot_prod >= 0.0f) {
					done = true;
					// printf("Error case 2\n");
				}

				int next_face_cnt = 0;
				for (int i = 0; i < cnt; i++) {
					int curr_tet_other_face_id = curr_tet_other_faces[i];
					int curr_tet_other_face_vert_id0 = faces[curr_tet_other_face_id][0];
					int curr_tet_other_face_vert_id1 = faces[curr_tet_other_face_id][1];
					int curr_tet_other_face_vert_id2 = faces[curr_tet_other_face_id][2];

					float3 p0 = { verts[curr_tet_other_face_vert_id0][0], verts[curr_tet_other_face_vert_id0][1], verts[curr_tet_other_face_vert_id0][2] };
					float3 p1 = { verts[curr_tet_other_face_vert_id1][0], verts[curr_tet_other_face_vert_id1][1], verts[curr_tet_other_face_vert_id1][2] };
					float3 p2 = { verts[curr_tet_other_face_vert_id2][0], verts[curr_tet_other_face_vert_id2][1], verts[curr_tet_other_face_vert_id2][2] };
					
					float3 curr_tet_other_face_tuv;
					float3 curr_tet_other_face_outward_normal;
					bool not_edge_case = ray_tri_intersection(
						this_ray_o,
						this_ray_d,
						p0,
						p1,
						p2,
						curr_tet_other_face_tuv
					);

					if (!not_edge_case)
						continue;

					bool curr_tet_other_face_intersect = (
						curr_tet_other_face_tuv.x >= 0.0f && 
						curr_tet_other_face_tuv.y >= 0.0f && 
						curr_tet_other_face_tuv.z >= 0.0f && 
						curr_tet_other_face_tuv.y + curr_tet_other_face_tuv.z <= 1.0f
					);

					tet_face_outward_normal(
						verts,
						faces,
						tets,
						curr_tet_other_face_id,
						curr_tet,
						curr_tet_other_face_outward_normal
					);
					float curr_tet_other_face_normal_dot_prod = dot(curr_tet_other_face_outward_normal, this_ray_d);

					if (curr_tet_other_face_intersect && curr_tet_other_face_normal_dot_prod > 0.0f) {
						next_face = curr_tet_other_faces[i];
						next_rt = curr_tet_other_face_tuv.x;
						next_iu = curr_tet_other_face_tuv.y;
						next_iv = curr_tet_other_face_tuv.z;
						next_face_cnt++;
					}
				}

				// In edge case, there could be multiple intersecting faces,
				// but generally, there should be only one intersecting face.
				if (next_face_cnt != 1) {
					// it should not happen, but we can't believe numerics...
					done = true;
					// printf("Error case 3\n");
				}
				else {
					for (int i = 0; i < 2; i++) {
						int p_next_tet = face_tets[next_face][i];
						if (p_next_tet == curr_tet)
							continue;
						next_tet = p_next_tet;
						break;
					}
				}

				curr_face = next_face;
				curr_tet = next_tet;
				curr_rt = next_rt;
				curr_iu = next_iu;
				curr_iv = next_iv;
			}
		}
		render_layers_cnt[batch_id][pix.y][pix.x] = num_layers_done;
	}


	void generate_render_layers(
		const dim3 grid, dim3 block,
		const torch::Tensor& verts,
		const torch::Tensor& faces,
		const torch::Tensor& tets,
		const torch::Tensor& face_tets,
		const torch::Tensor& tet_faces,
		const torch::Tensor& faces_existence,
		
		const torch::Tensor& ray_o,
		const torch::Tensor& ray_d,

		const int* first_face,
		const int* first_tet,

		int W, int H,
		int P, int F,
		
		int num_layers,
		torch::Tensor& render_layers,
		torch::Tensor& render_layers_cnt
	) {
		generateRenderLayersCUDA<< <grid, block >> > (
			verts.packed_accessor64<float, 2>(),
			faces.packed_accessor64<int, 2>(),
			tets.packed_accessor64<int, 2>(),
			face_tets.packed_accessor64<int, 2>(),
			tet_faces.packed_accessor64<int, 2>(),
			faces_existence.packed_accessor64<int, 1>(),

			ray_o.packed_accessor64<float, 4>(),
			ray_d.packed_accessor64<float, 4>(),

			first_face,
			first_tet,

			W, H,
			P, F,

			num_layers,
			render_layers.packed_accessor64<int, 4>(),
			render_layers_cnt.packed_accessor64<int, 3>()
		);
	}
}