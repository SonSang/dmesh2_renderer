#ifndef CUDA_FORWARD_H_INCLUDED
#define CUDA_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

namespace FORWARD
{
	/*
	==============================================
	3D renderer (approximate depth testing)
	==============================================
	*/
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
	);

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
	);

    /*
    ==============================================
    Render Layer Generator (precise depth testing based on tet grid) (non-differentiable)
    ==============================================
    */
   	// Find the first intersecting face between each pixel ray.
	void first_intersect(
		const dim3 grid, dim3 block,
		const torch::Tensor& verts,
		const torch::Tensor& faces,
		const torch::Tensor& tets,
		const torch::Tensor& face_tets,
		const float* faces_min_depth,
		const float* faces_max_depth,
		const uint2* ranges,
		const uint32_t* face_list,
		int B, int F, int W, int H,
		const torch::Tensor& ray_o,
		const torch::Tensor& ray_d,
		int* first_face,
		int* first_tet
	);

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
	);
}


#endif