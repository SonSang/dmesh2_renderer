#ifndef CUDA_BACKWARD_H_INCLUDED
#define CUDA_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

namespace BACKWARD
{
	/*
	==============================================
	3D renderer
	==============================================
	*/
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
		const torch::Tensor& faces_opacity,				// (F)

		const torch::Tensor& verts_ndc,					// (B, P, 3)
		const torch::Tensor& verts_image,				// (B, P, 2)
		const torch::Tensor& faces_intense,				// (B, F)

		const float aa_temperature,
		const torch::Tensor& aa_face_verts,				// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges,				// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_iszero,		// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_recip,		// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_normal,		// (B, F, 3, 2)
		const torch::Tensor& aa_face_edges_normal_c,	// (B, F, 3)

		const torch::Tensor& image_ray_o,				// (B, H, W, 3)
		const torch::Tensor& image_ray_d,				// (B, H, W, 3)

		// incoming gradient
		const torch::Tensor& dL_dout_color, 			// (B, H, W, 3)
		const torch::Tensor& dL_dout_depth, 			// (B, H, W)

		// buffers
		const uint2* ranges,
		const float* final_Ts,
		const float* final_prev_Ts,
		const uint32_t* n_contrib,
		const uint32_t* face_list,
		
		int len_oarea_buffer,									// K
		const torch::Tensor& oarea_buffer_oarea,				// (B, H, W, K)
		const torch::Tensor& oarea_buffer_tri_id,				// (B, H, W, K)
		const torch::Tensor& oarea_buffer_tri_cnt,				// (B, H, W)
		const torch::Tensor& oarea_buffer_doarea_dtri_verts,	// (B, H, W, K, 3, 2)

		// outgoing gradient
		torch::Tensor& dL_dverts,						// (P, 3)
		torch::Tensor& dL_dverts_color,					// (P, 3)
		torch::Tensor& dL_dfaces_opacity,				// (F)
		torch::Tensor& dL_dverts_ndc,					// (B, P, 3)
		torch::Tensor& dL_dfaces_intense,				// (B, F)
		torch::Tensor& dL_daa_face_verts				// (B, F, 3, 2)
	);
}


#endif