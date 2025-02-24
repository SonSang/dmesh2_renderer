#ifndef CUDA_RENDERER_H_INCLUDED
#define CUDA_RENDERER_H_INCLUDED

#include <vector>
#include <functional>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

namespace CudaRenderer
{
    /*
    ==============================================
    3D Renderer (approximate depth testing)
    ==============================================
    */
    class Renderer
    {
    public:
		static int forward(
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
        );

		static void backward(
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
        );
    };

    /*
    ==============================================
    Render Layer Generator (precise depth testing based on tet grid) (non-differentiable)
    ==============================================
    */
   	class RenderLayerGenerator
    {
    public:
		static void forward(
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
		);
    };
}

#endif