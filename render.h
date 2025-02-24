#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

/*
============================================================
3D
============================================================
*/
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderForwardCUDA(
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
    int len_grad_oarea_tri_verts,						// K

    // ray
    const torch::Tensor& image_ray_o,					// (B, H, W, 3)
    const torch::Tensor& image_ray_d					// (B, H, W, 3)
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderBackwardCUDA(
    int num_rendered,
    const torch::Tensor& background,					// (3)                          no grad
			
    // render subset of the image
    const torch::Tensor& patch_min,						// (B, 2)                       no grad
    const int patch_width,
    const int patch_height,
    
    // global info (shared across all batches)
    const torch::Tensor& verts,							// (P, 3)                       grad
    const torch::Tensor& faces,							// (F, 3)                       no grad
    const torch::Tensor& verts_color,					// (P, 3)                       grad
    const torch::Tensor& faces_opacity,					// (F)                          grad

    // local info (per batch)
    const torch::Tensor& verts_ndc,						// (B, P, 3)                    grad (only last channel for depth)
    const torch::Tensor& verts_image,					// (B, P, 2)                    no grad
    const torch::Tensor& faces_intense,					// (B, F)                       grad

    // for aa
    float aa_temperature,
    const torch::Tensor& aa_face_verts,					// (B, F, 3, 2)                 grad
    const torch::Tensor& aa_face_edges,					// (B, F, 3, 2)                 no grad
    const torch::Tensor& aa_face_edges_iszero,			// (B, F, 3, 2)                 no grad
    const torch::Tensor& aa_face_edges_recip,			// (B, F, 3, 2)                 no grad
    const torch::Tensor& aa_face_edges_normal,			// (B, F, 3, 2)                 no grad
    const torch::Tensor& aa_face_edges_normal_c,		// (B, F, 3)                    no grad
    int len_oarea_buffer,						        // K

    // ray
    const torch::Tensor& image_ray_o,					// (B, H, W, 3)                 no grad
    const torch::Tensor& image_ray_d,					// (B, H, W, 3)                 no grad

    // incoming gradient
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,

    // buffers
    const torch::Tensor& face_buffer,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& image_buffer,
    const torch::Tensor& oarea_buffer_oarea,			// (B, H, W, K)                 no grad
    const torch::Tensor& oarea_buffer_tri_id,			// (B, H, W, K)                 no grad
    const torch::Tensor& oarea_buffer_tri_cnt,			// (B, H, W)                    no grad
    const torch::Tensor& oarea_buffer_doarea_dtri_verts	// (B, H, W, K, 3, 2)           no grad
);

/*
============================================================
Render layers
============================================================
*/
std::tuple<torch::Tensor, torch::Tensor>
GenerateRenderLayersCUDA(
    int width, int height,

    const torch::Tensor& verts,                         // (P, 3) 
    const torch::Tensor& faces,                         // (F, 3)
    const torch::Tensor& tets,                          // (T, 4)
    const torch::Tensor& face_tets,                     // (F, 2)
    const torch::Tensor& tet_faces,                     // (T, 4)
    const torch::Tensor& face_existence,                // (F)

    const torch::Tensor& verts_ndc,                     // (B, P, 3)
    const torch::Tensor& verts_image,                   // (B, P, 2)

    const torch::Tensor& image_ray_o,                   // (B, H, W, 3)
    const torch::Tensor& image_ray_d,                   // (B, H, W, 3)

    int num_layers                                     // L
);