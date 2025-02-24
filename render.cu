#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include "cuda_impl/renderer.h"

/*
============================================================
3D
============================================================
*/
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    int len_oarea_buffer,						        // K

    // ray
    const torch::Tensor& image_ray_o,					// (B, H, W, 3)
    const torch::Tensor& image_ray_d					// (B, H, W, 3)
) {
    if (background.ndimension() != 1 || background.size(0) != 3) {
        AT_ERROR("background must have dimensions (3,)");
    }
    if (patch_min.ndimension() != 2 || patch_min.size(1) != 2) {
        AT_ERROR("patch_min must have dimensions (B, 2)");
    }
    if (verts.ndimension() != 2 || verts.size(1) != 3) {
        AT_ERROR("verts must have dimensions (P, 3)");
    }
    if (faces.ndimension() != 2 || faces.size(1) != 3) {
        AT_ERROR("faces must have dimensions (F, 3)");
    }
    if (verts_color.ndimension() != 2 || verts_color.size(1) != 3) {
        AT_ERROR("vert color must have dimensions (P, 3)");
    }
    if (faces_opacity.ndimension() != 1 || faces_opacity.size(0) != faces.size(0)) {
        AT_ERROR("face opacity must have dimensions (F,)");
    }
    if (verts_ndc.ndimension() != 3 || verts_ndc.size(2) != 3) {
        AT_ERROR("verts_ndc must have dimensions (B, P, 3)");
    }
    if (verts_image.ndimension() != 3 || verts_image.size(2) != 2) {
        AT_ERROR("verts_image must have dimensions (B, P, 2)");
    }
    if (faces_intense.ndimension() != 2 || faces_intense.size(1) != faces.size(0)) {
        AT_ERROR("faces_intense must have dimensions (B, F,)");
    }
    if (aa_face_verts.ndimension() != 4 || aa_face_verts.size(2) != 3 || aa_face_verts.size(3) != 2) {
        AT_ERROR("aa_face_verts must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges.ndimension() != 4 || aa_face_edges.size(2) != 3 || aa_face_edges.size(3) != 2) {
        AT_ERROR("aa_face_edges must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_iszero.ndimension() != 4 || aa_face_edges_iszero.size(2) != 3 || aa_face_edges_iszero.size(3) != 2) {
        AT_ERROR("aa_face_edges_iszero must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_recip.ndimension() != 4 || aa_face_edges_recip.size(2) != 3 || aa_face_edges_recip.size(3) != 2) {
        AT_ERROR("aa_face_edges_recip must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_normal.ndimension() != 4 || aa_face_edges_normal.size(2) != 3 || aa_face_edges_normal.size(3) != 2) {
        AT_ERROR("aa_face_edges_normal must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_normal_c.ndimension() != 3 || aa_face_edges_normal_c.size(2) != 3) {
        AT_ERROR("aa_face_edges_normal_c must have dimensions (B, F, 3)");
    }
    if (image_ray_o.ndimension() != 4 || image_ray_o.size(3) != 3) {
        AT_ERROR("image_ray_o must have dimensions (B, H, W, 3)");
    }
    if (image_ray_d.ndimension() != 4 || image_ray_d.size(3) != 3) {
        AT_ERROR("image_ray_d must have dimensions (B, H, W, 3)");
    }
    if (aa_temperature < 0 || aa_temperature > 1) {
        AT_ERROR("aa_temperature must be in the range [0, 1]");
    }
    if (len_oarea_buffer < 0) {
        AT_ERROR("len_oarea_buffer must be non-negative");
    }

    const int B = verts_ndc.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int H = patch_height;
    const int W = patch_width;

    auto float_opts = verts.options().dtype(torch::kFloat32);
    auto int_opts = verts.options().dtype(torch::kInt32);
    torch::Tensor out_color = torch::full({B, H, W, 3}, 0.0, float_opts);
    torch::Tensor out_depth = torch::full({B, H, W}, 0.0, float_opts);
    
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kByte);
    torch::Tensor face_buffer = torch::empty({0}, options.device(device));
    torch::Tensor binning_buffer = torch::empty({0}, options.device(device));
    torch::Tensor img_buffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> faceFunc = resizeFunctional(face_buffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binning_buffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(img_buffer);

    // aa grads
    if (aa_temperature == 0.0f)
        len_oarea_buffer = 0;           // if temperature is 0, we do not need to compute oarea
    torch::Tensor oarea_buffer_oarea = torch::zeros({B, H, W, len_oarea_buffer}, float_opts);
    torch::Tensor oarea_buffer_tri_id = torch::zeros({B, H, W, len_oarea_buffer}, int_opts);
    torch::Tensor oarea_buffer_tri_cnt = torch::zeros({B, H, W}, int_opts);
    torch::Tensor oarea_buffer_doarea_dtri_verts = torch::zeros({B, H, W, len_oarea_buffer, 3, 2}, float_opts);
    
    int rendered = 0;
    if(P != 0)
    {
        rendered = CudaRenderer::Renderer::forward(
            faceFunc,
            binningFunc,
            imgFunc,
            
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
            
            out_color,
            out_depth,
            
            len_oarea_buffer,
            oarea_buffer_oarea,
            oarea_buffer_tri_id,
            oarea_buffer_tri_cnt,
            oarea_buffer_doarea_dtri_verts
        );
    }

    return std::make_tuple(rendered, out_color, out_depth, oarea_buffer_oarea, oarea_buffer_tri_id, oarea_buffer_tri_cnt, oarea_buffer_doarea_dtri_verts, face_buffer, binning_buffer, img_buffer);
}


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

) {
    if (background.ndimension() != 1 || background.size(0) != 3) {
        AT_ERROR("background must have dimensions (3,)");
    }
    if (patch_min.ndimension() != 2 || patch_min.size(1) != 2) {
        AT_ERROR("patch_min must have dimensions (B, 2)");
    }
    if (verts.ndimension() != 2 || verts.size(1) != 3) {
        AT_ERROR("verts must have dimensions (P, 3)");
    }
    if (faces.ndimension() != 2 || faces.size(1) != 3) {
        AT_ERROR("faces must have dimensions (F, 3)");
    }
    if (verts_color.ndimension() != 2 || verts_color.size(1) != 3) {
        AT_ERROR("vert color must have dimensions (P, 3)");
    }
    if (faces_opacity.ndimension() != 1 || faces_opacity.size(0) != faces.size(0)) {
        AT_ERROR("face opacity must have dimensions (F,)");
    }
    if (verts_ndc.ndimension() != 3 || verts_ndc.size(2) != 3) {
        AT_ERROR("verts_ndc must have dimensions (B, P, 3)");
    }
    if (verts_image.ndimension() != 3 || verts_image.size(2) != 2) {
        AT_ERROR("verts_image must have dimensions (B, P, 2)");
    }
    if (faces_intense.ndimension() != 2 || faces_intense.size(1) != faces.size(0)) {
        AT_ERROR("faces_intense must have dimensions (B, F,)");
    }
    if (aa_face_verts.ndimension() != 4 || aa_face_verts.size(2) != 3 || aa_face_verts.size(3) != 2) {
        AT_ERROR("aa_face_verts must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges.ndimension() != 4 || aa_face_edges.size(2) != 3 || aa_face_edges.size(3) != 2) {
        AT_ERROR("aa_face_edges must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_iszero.ndimension() != 4 || aa_face_edges_iszero.size(2) != 3 || aa_face_edges_iszero.size(3) != 2) {
        AT_ERROR("aa_face_edges_iszero must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_recip.ndimension() != 4 || aa_face_edges_recip.size(2) != 3 || aa_face_edges_recip.size(3) != 2) {
        AT_ERROR("aa_face_edges_recip must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_normal.ndimension() != 4 || aa_face_edges_normal.size(2) != 3 || aa_face_edges_normal.size(3) != 2) {
        AT_ERROR("aa_face_edges_normal must have dimensions (B, F, 3, 2)");
    }
    if (aa_face_edges_normal_c.ndimension() != 3 || aa_face_edges_normal_c.size(2) != 3) {
        AT_ERROR("aa_face_edges_normal_c must have dimensions (B, F, 3)");
    }
    if (image_ray_o.ndimension() != 4 || image_ray_o.size(3) != 3) {
        AT_ERROR("image_ray_o must have dimensions (B, H, W, 3)");
    }
    if (image_ray_d.ndimension() != 4 || image_ray_d.size(3) != 3) {
        AT_ERROR("image_ray_d must have dimensions (B, H, W, 3)");
    }
    if (aa_temperature < 0 || aa_temperature > 1) {
        AT_ERROR("aa_temperature must be in the range [0, 1]");
    }
    if (len_oarea_buffer < 0) {
        AT_ERROR("len_oarea_buffer must be non-negative");
    }

    const int B = verts_ndc.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int H = patch_height;
    const int W = patch_width;

    auto float_opts = verts.options().dtype(torch::kFloat32);
    auto int_opts = verts.options().dtype(torch::kInt32);
    torch::Tensor dL_dverts = torch::zeros_like(verts);
    torch::Tensor dL_dverts_color = torch::zeros_like(verts_color);
    torch::Tensor dL_dfaces_opacity = torch::zeros_like(faces_opacity);
    torch::Tensor dL_dverts_ndc = torch::zeros_like(verts_ndc);
    torch::Tensor dL_dfaces_intense = torch::zeros_like(faces_intense);
    torch::Tensor dL_daa_face_verts = torch::zeros_like(aa_face_verts);

    if(F != 0)
    {  
        CudaRenderer::Renderer::backward(
            B, P, F, num_rendered,
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

            dL_dout_color,
            dL_dout_depth,

            reinterpret_cast<char*>(face_buffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binning_buffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(image_buffer.contiguous().data_ptr()),

            len_oarea_buffer,
            oarea_buffer_oarea,
            oarea_buffer_tri_id,
            oarea_buffer_tri_cnt,
            oarea_buffer_doarea_dtri_verts,
            
            dL_dverts,
            dL_dverts_color,
            dL_dfaces_opacity,
            dL_dverts_ndc,
            dL_dfaces_intense,
            dL_daa_face_verts
        );
    }

    return std::make_tuple(dL_dverts, dL_dverts_color, dL_dfaces_opacity, dL_dverts_ndc, dL_dfaces_intense, dL_daa_face_verts);
}

/*
Render Layers
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

    int num_layers                                      // L
) {
    if (verts.ndimension() != 2 || verts.size(1) != 3) {
        AT_ERROR("verts must have dimensions (P, 3)");
    }
    if (faces.ndimension() != 2 || faces.size(1) != 3) {
        AT_ERROR("faces must have dimensions (F, 3)");
    }
    if (tets.ndimension() != 2 || tets.size(1) != 4) {
        AT_ERROR("tets must have dimensions (T, 4)");
    }
    if (face_tets.ndimension() != 2 || face_tets.size(1) != 2) {
        AT_ERROR("face_tets must have dimensions (F, 2)");
    }
    if (tet_faces.ndimension() != 2 || tet_faces.size(1) != 4) {
        AT_ERROR("tet_faces must have dimensions (T, 4)");
    }
    if (face_existence.ndimension() != 1 || face_existence.size(0) != faces.size(0)) {
        AT_ERROR("face_existence must have dimensions (F,)");
    }
    if (verts_ndc.ndimension() != 3 || verts_ndc.size(2) != 3) {
        AT_ERROR("verts_ndc must have dimensions (B, P, 3)");
    }
    if (verts_image.ndimension() != 3 || verts_image.size(2) != 2) {
        AT_ERROR("verts_image must have dimensions (B, P, 2)");
    }
    if (image_ray_o.ndimension() != 4 || image_ray_o.size(3) != 3) {
        AT_ERROR("image_ray_o must have dimensions (B, H, W, 3)");
    }
    if (image_ray_d.ndimension() != 4 || image_ray_d.size(3) != 3) {
        AT_ERROR("image_ray_d must have dimensions (B, H, W, 3)");
    }
    if (num_layers < 0) {
        AT_ERROR("num_layers must be non-negative");
    }

    const int B = verts_ndc.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int T = tets.size(0);
    
    auto int_opts = verts.options().dtype(torch::kInt32);
    torch::Tensor render_layers_cnt = torch::zeros({B, height, width}, int_opts);
    torch::Tensor render_layers = torch::full({B, height, width, num_layers}, -1, int_opts);

    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kByte);
    torch::Tensor face_buffer = torch::empty({0}, options.device(device));
    torch::Tensor binning_buffer = torch::empty({0}, options.device(device));
    torch::Tensor img_buffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> faceFunc = resizeFunctional(face_buffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binning_buffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(img_buffer);

    CudaRenderer::RenderLayerGenerator::forward(
        faceFunc,
        binningFunc,
        imgFunc,
        
        B, P, F, T,
        width, height,

        verts,
        faces,
        tets,
        face_tets,
        tet_faces,
        face_existence,

        verts_ndc,
        verts_image,

        image_ray_o,
        image_ray_d,
        
        num_layers,
        render_layers,
        render_layers_cnt        
    );

    return std::make_tuple(render_layers, render_layers_cnt);
}