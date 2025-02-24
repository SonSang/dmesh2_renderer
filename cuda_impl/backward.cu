#include "forward.h"
#include "auxiliary.h"
#include "cuda_math.h"
#include "aa.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <tuple>

namespace BACKWARD {

    /*
	========================================================
	Rendering (3D)
	========================================================
	*/
    template <uint32_t CHANNELS>
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	renderCUDA(
        const int B, int P, int F,
        torch::PackedTensorAccessor64<float, 1> background,             // (3)

        torch::PackedTensorAccessor64<int, 2> patch_min,                // (B, 2)
        const int patch_width,
        const int patch_height,

        torch::PackedTensorAccessor64<float, 2> verts,                  // (P, 3)
        torch::PackedTensorAccessor64<int, 2> faces,                    // (F, 3)
        torch::PackedTensorAccessor64<float, 2> verts_color,            // (P, 3)
        torch::PackedTensorAccessor64<float, 1> faces_opacity,          // (F)

        torch::PackedTensorAccessor64<float, 3> verts_ndc,              // (B, P, 3)
        torch::PackedTensorAccessor64<float, 3> verts_image,            // (B, P, 2)
        torch::PackedTensorAccessor64<float, 2> faces_intense,          // (B, F)

        const float aa_temperature,
        torch::PackedTensorAccessor64<float, 4> aa_face_verts,          // (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 4> aa_face_edges,          // (B, F, 3, 2)
        torch::PackedTensorAccessor64<bool, 4> aa_face_edges_iszero,    // (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 4> aa_face_edges_recip,    // (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 4> aa_face_edges_normal,   // (B, F, 3, 2)
        torch::PackedTensorAccessor64<float, 3> aa_face_edges_normal_c, // (B, F, 3)

        torch::PackedTensorAccessor64<float, 3> aa_face_verts_min,          // (B, F, 2)
        torch::PackedTensorAccessor64<float, 3> aa_face_verts_max,          // (B, F, 2)

        torch::PackedTensorAccessor64<float, 4> ray_o,            // (B, H, W, 3)
        torch::PackedTensorAccessor64<float, 4> ray_d,            // (B, H, W, 3)

        torch::PackedTensorAccessor64<float, 4> dL_dout_color,          // (B, H, W, 3)
        torch::PackedTensorAccessor64<float, 3> dL_dout_depth,          // (B, H, W)

        const uint2* __restrict__ ranges,
        const float* __restrict__ final_Ts,
        const float* __restrict__ final_prev_Ts,
        const uint32_t* __restrict__ n_contrib,
        const uint32_t* __restrict__ face_list,

        int len_oarea_buffer,
        torch::PackedTensorAccessor64<float, 4> oarea_buffer_oarea,                 // (B, H, W, K)
        torch::PackedTensorAccessor64<int, 4> oarea_buffer_tri_id,                  // (B, H, W, K)
        torch::PackedTensorAccessor64<int, 3> oarea_buffer_tri_cnt,                 // (B, H, W)
        torch::PackedTensorAccessor64<float, 6> oarea_buffer_doarea_dtri_verts,     // (B, H, W, K, 3, 2)

        torch::PackedTensorAccessor64<float, 2> dL_dverts,              // (P, 3)
        torch::PackedTensorAccessor64<float, 2> dL_dverts_color,        // (P, 3)
        torch::PackedTensorAccessor64<float, 1> dL_dfaces_opacity,      // (F)
        torch::PackedTensorAccessor64<float, 3> dL_dverts_ndc,          // (B, P, 3)
        torch::PackedTensorAccessor64<float, 2> dL_dfaces_intense,      // (B, F)
        torch::PackedTensorAccessor64<float, 4> dL_daa_face_verts       // (B, F, 3, 2)
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

		__shared__ int collected_face_vert_id_0[BLOCK_SIZE];
		__shared__ int collected_face_vert_id_1[BLOCK_SIZE];
		__shared__ int collected_face_vert_id_2[BLOCK_SIZE];

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

		// In the forward, we stored the final value for T, the
		// product of all (1 - alpha) factors. 
		const float T_final = inside ? final_Ts[b_pix_id] : 0;
		const float prev_T_final = inside ? final_prev_Ts[b_pix_id] : 0;
		float T = prev_T_final;
		bool T_first_pass = true;

		// We start from the back. The ID of the last contributing
		// face is known from each pixel from the forward.
		uint32_t contributor = toDo;
		const int last_contributor = inside ? n_contrib[b_pix_id] : 0;

		float accum_rec[CHANNELS] = { 0 };
		float accum_recd[1] = { 0 };
		float dL_dpixel_color[CHANNELS];
		float dL_dpixel_depth[1];
		if (inside) {
			for (int i = 0; i < CHANNELS; i++)
				dL_dpixel_color[i] = dL_dout_color[batch_id][pix.y][pix.x][i];
            dL_dpixel_depth[0] = dL_dout_depth[batch_id][pix.y][pix.x];
		}

        // buffer pointer into oarea buffer
        int oarea_buffer_pointer = oarea_buffer_tri_cnt[batch_id][pix.y][pix.x];
			
		float last_alpha = 0;
		float last_color[CHANNELS] = { 0 };
		float last_depth[1] = { 0 }; 
		
		// Traverse all Faces
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// Load auxiliary data into shared memory, start in the BACK
			// and load them in reverse order.
			block.sync();
			const int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				const int face_id = face_list[range.y - progress - 1];	// start from BACK

				collected_face_id[block.thread_rank()] = face_id;

				int face_vert_id_0 = faces[face_id][0];
                int face_vert_id_1 = faces[face_id][1];
                int face_vert_id_2 = faces[face_id][2];
                
				collected_face_vert_id_0[block.thread_rank()] = face_vert_id_0;
				collected_face_vert_id_1[block.thread_rank()] = face_vert_id_1;
				collected_face_vert_id_2[block.thread_rank()] = face_vert_id_2;

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
                
				collected_face_opacity[block.thread_rank()] = faces_opacity[face_id];
                collected_face_intense[block.thread_rank()] = faces_intense[batch_id][face_id];
			}
			block.sync();

			// Iterate over faces
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				// Keep track of current face ID. Skip, if this one
				// is behind the last contributor for this pixel.
				contributor--;
				if (contributor >= last_contributor)
					continue;

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

                // Based on the gradient buffer, find out if there was an intersection between the current face and the current pixel
                float oarea = 0.0f;
                float doarea_dtri_verts[3][2] = { { 0, 0 }, { 0, 0 }, { 0, 0 } };
                int oarea_error_code = 0;

                bool need_to_compute_oarea = false;
				if (aa_temperature > 0.0f) {
					if (oarea_buffer_pointer > 0) {
						int tri_id_buf = oarea_buffer_tri_id[batch_id][pix.y][pix.x][oarea_buffer_pointer - 1];
						if (tri_id_buf == face_id) {
							// there was non-zero overlapping area between the current face and the current pixel
							oarea = oarea_buffer_oarea[batch_id][pix.y][pix.x][oarea_buffer_pointer - 1];
							for (int ii = 0; ii < 3; ii++) 
								for (int jj = 0; jj < 2; jj++) 
									doarea_dtri_verts[ii][jj] = oarea_buffer_doarea_dtri_verts[batch_id][pix.y][pix.x][oarea_buffer_pointer - 1][ii][jj];
							oarea_buffer_pointer--;
						}
						else {
							if (oarea_buffer_pointer == len_oarea_buffer) {
								// there could have been buffer overflow
								need_to_compute_oarea = true;
							}
							else {
								// there was no intersection between the current face and the current pixel
								need_to_compute_oarea = false;
							}
						}
					}
					else {
						if (len_oarea_buffer > 0) {
							// there was no intersection between the current face and the current pixel
							need_to_compute_oarea = false;
						}
						else {
							// there was no buffer at all
							need_to_compute_oarea = true;
						}
					}
				}

                if (need_to_compute_oarea) {
                    oarea_error_code = AA::tri_pix_overlap_area(
                        aa_face_verts, aa_face_edges, aa_face_edges_iszero, aa_face_edges_recip, aa_face_edges_normal, aa_face_edges_normal_c,
                        batch_id, face_id,
                        txmin, txmax, tymin, tymax,
                        pxmin, pxmax, pymin, pymax, pix_area,
                        &oarea, doarea_dtri_verts
                    );
                }

				if (aa_temperature > 0.0f) {
					if ((oarea_error_code != 0) || (oarea == 0.0f))
						continue;
				}
					
                float oarea_ratio = oarea / pix_area;

				// Find intersection point between ray and triangle as forward;
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
				
				// Recover previous T
				// at first pass, T is already prev T before the end;
				if (!T_first_pass) {
					// (1.0 - alpha) cannot be 0.0, because if it was,
					// it would have been dealt with in the first pass;
					T = T / (1.f - alpha);
					if (isnan(T) || isinf(T)) {
						printf("[BACKWARD]: T is nan or inf, which should not happen\n");
					}
				}
				T_first_pass = false;

				/*
				Compute gradients of iC, iD and alpha w.r.t. loss.
				*/
				float curr_dL_dicolor[CHANNELS] = { 0 };
				float curr_dL_didepth[1] = { 0 };
				float curr_dL_dalpha = 0.0f;
				
				// color
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					const float c = iC[ch];
					// Update last color (to be used in the next iteration)
					accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;

					const float dL_dchannel = dL_dpixel_color[ch];
					curr_dL_dicolor[ch] = dL_dchannel * alpha * T;
					curr_dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				}

				// depth;
				for (int ch = 0; ch < 1; ch++) {
					const float c = iD[ch];
					// Update last depth (to be used in the next iteration)
					accum_recd[ch] = last_alpha * last_depth[ch] + (1.f - last_alpha) * accum_recd[ch];
					last_depth[ch] = c;

					const float dL_dchannel = dL_dpixel_depth[ch];
					curr_dL_didepth[ch] = dL_dchannel * alpha * T;
					curr_dL_dalpha += (c - accum_recd[ch]) * dL_dchannel;
				}

				// alpha: additional contribution from the background;
				curr_dL_dalpha *= T;
				// Update last alpha (to be used in the next iteration)
				last_alpha = alpha;

				// Account for fact that alpha also influences how much of
				// the background color is added if nothing left to blend
				float bg_dot_dpixel = 0;
				float bd_dot_dpixel = 0;
				for (int ch = 0; ch < CHANNELS; ch++)
					bg_dot_dpixel += background[ch] * dL_dpixel_color[ch];
				for (int ch = 0; ch < 1; ch++)
					bd_dot_dpixel += 1.0 * dL_dpixel_depth[ch];
				
                if (alpha == 1.0f) {
					// in this case, (-T_final / (1.f - alpha)) is (-prev_T_final),
					// because when alpha == 1.0, it would have been the last step;
					curr_dL_dalpha += (-prev_T_final) * bg_dot_dpixel;
					curr_dL_dalpha += (-prev_T_final) * bd_dot_dpixel;
				}
				else {
					curr_dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
					curr_dL_dalpha += (-T_final / (1.f - alpha)) * bd_dot_dpixel;
				}

                // Compute gradients of faces opacity, aa face verts
                float curr_dL_dfopacity = curr_dL_dalpha * oarea_ratio;
                float curr_dL_doarea_ratio = (curr_dL_dalpha * collected_face_opacity[j]) * aa_temperature;
                float curr_dL_doarea = curr_dL_doarea_ratio / pix_area;

                float curr_dL_daa_face_verts[3][2] = { { 0, 0 }, { 0, 0 }, { 0, 0 } };
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        curr_dL_daa_face_verts[ii][jj] = curr_dL_doarea * doarea_dtri_verts[ii][jj];
                        
				/*
				Compute gradients of (i0, i1, i2), verts color, faces opacity, faces intense, faces normal.
				*/
				float curr_dL_di0 = 0, curr_dL_di1 = 0, curr_dL_di2 = 0;
				
				float curr_dL_dvcolor_0[CHANNELS] = { 0 };
				float curr_dL_dvcolor_1[CHANNELS] = { 0 };
				float curr_dL_dvcolor_2[CHANNELS] = { 0 };
				
				float curr_dL_dvdepth_0[1] = { 0 };
				float curr_dL_dvdepth_1[1] = { 0 };
				float curr_dL_dvdepth_2[1] = { 0 };

				float curr_dL_dfintense = 0;
				
				// color
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					curr_dL_di0 += collected_face_vert_color_0[j * CHANNELS + ch] * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_di1 += collected_face_vert_color_1[j * CHANNELS + ch] * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_di2 += collected_face_vert_color_2[j * CHANNELS + ch] * curr_dL_dicolor[ch] * collected_face_intense[j];

					curr_dL_dvcolor_0[ch] += i0 * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_dvcolor_1[ch] += i1 * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_dvcolor_2[ch] += i2 * curr_dL_dicolor[ch] * collected_face_intense[j];

					curr_dL_dfintense += (i0 * collected_face_vert_color_0[j * CHANNELS + ch] +
									i1 * collected_face_vert_color_1[j * CHANNELS + ch] +
									i2 * collected_face_vert_color_2[j * CHANNELS + ch]) * curr_dL_dicolor[ch];
				}

				// depth
				curr_dL_di0 += collected_face_vert_depth_0[j] * curr_dL_didepth[0];
				curr_dL_di1 += collected_face_vert_depth_1[j] * curr_dL_didepth[0];
				curr_dL_di2 += collected_face_vert_depth_2[j] * curr_dL_didepth[0];

				curr_dL_dvdepth_0[0] += i0 * curr_dL_didepth[0];
				curr_dL_dvdepth_1[0] += i1 * curr_dL_didepth[0];
				curr_dL_dvdepth_2[0] += i2 * curr_dL_didepth[0];

				/*
				Compute gradients of verts w.r.t. loss.
				*/
				float di0_diuc = -1, di0_divc = -1;
				float di1_diuc = 1, di1_divc = 0;
				float di2_diuc = 0, di2_divc = 1;

				float diuc_diu, diuc_div, divc_diu, divc_div;
				clamp_bary_uv_grad(iclamp_code, diuc_diu, diuc_div, divc_diu, divc_div);

				float di0_diu = di0_diuc * diuc_diu + di0_divc * divc_diu;
				float di0_div = di0_diuc * diuc_div + di0_divc * divc_div;
				float di1_diu = di1_diuc * diuc_diu + di1_divc * divc_diu;
				float di1_div = di1_diuc * diuc_div + di1_divc * divc_div;
				float di2_diu = di2_diuc * diuc_diu + di2_divc * divc_diu;
				float di2_div = di2_diuc * diuc_div + di2_divc * divc_div;

				float curr_dL_diu = curr_dL_di0 * di0_diu + curr_dL_di1 * di1_diu + curr_dL_di2 * di2_diu;
				float curr_dL_div = curr_dL_di0 * di0_div + curr_dL_di1 * di1_div + curr_dL_di2 * di2_div;

				float3 curr_diu_dp0, curr_diu_dp1, curr_diu_dp2;
				float3 curr_div_dp0, curr_div_dp1, curr_div_dp2;
				ray_tri_intersection_grad(
					this_ray_o, 
					this_ray_d,
					v0, v1, v2,
					curr_diu_dp0, curr_diu_dp1, curr_diu_dp2,
					curr_div_dp0, curr_div_dp1, curr_div_dp2);

				float3 curr_dL_dp0 = curr_dL_diu * curr_diu_dp0 + curr_dL_div * curr_div_dp0;
				float3 curr_dL_dp1 = curr_dL_diu * curr_diu_dp1 + curr_dL_div * curr_div_dp1;
				float3 curr_dL_dp2 = curr_dL_diu * curr_diu_dp2 + curr_dL_div * curr_div_dp2;

				/*
				Aggregate gradients.
				*/

				// verts;
				atomicAdd(&(dL_dverts[collected_face_vert_id_0[j]][0]), curr_dL_dp0.x);
                atomicAdd(&(dL_dverts[collected_face_vert_id_0[j]][1]), curr_dL_dp0.y);
                atomicAdd(&(dL_dverts[collected_face_vert_id_0[j]][2]), curr_dL_dp0.z);
                
                atomicAdd(&(dL_dverts[collected_face_vert_id_1[j]][0]), curr_dL_dp1.x);
                atomicAdd(&(dL_dverts[collected_face_vert_id_1[j]][1]), curr_dL_dp1.y);
                atomicAdd(&(dL_dverts[collected_face_vert_id_1[j]][2]), curr_dL_dp1.z);
                
                atomicAdd(&(dL_dverts[collected_face_vert_id_2[j]][0]), curr_dL_dp2.x);
                atomicAdd(&(dL_dverts[collected_face_vert_id_2[j]][1]), curr_dL_dp2.y);
                atomicAdd(&(dL_dverts[collected_face_vert_id_2[j]][2]), curr_dL_dp2.z);
                
				// verts color;
				for (int ch = 0; ch < CHANNELS; ch++)
				{
                    atomicAdd(&(dL_dverts_color[collected_face_vert_id_0[j]][ch]), curr_dL_dvcolor_0[ch]);
                    atomicAdd(&(dL_dverts_color[collected_face_vert_id_1[j]][ch]), curr_dL_dvcolor_1[ch]);
                    atomicAdd(&(dL_dverts_color[collected_face_vert_id_2[j]][ch]), curr_dL_dvcolor_2[ch]);
				}

				// verts depth;
                atomicAdd(&(dL_dverts_ndc[batch_id][collected_face_vert_id_0[j]][2]), curr_dL_dvdepth_0[0]);
                atomicAdd(&(dL_dverts_ndc[batch_id][collected_face_vert_id_1[j]][2]), curr_dL_dvdepth_1[0]);
                atomicAdd(&(dL_dverts_ndc[batch_id][collected_face_vert_id_2[j]][2]), curr_dL_dvdepth_2[0]);

				// faces opacity;
				atomicAdd(&(dL_dfaces_opacity[collected_face_id[j]]), curr_dL_dfopacity);

				// faces intense;
				atomicAdd(&(dL_dfaces_intense[batch_id][collected_face_id[j]]), curr_dL_dfintense);

				// aa face verts;
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        atomicAdd(&(dL_daa_face_verts[batch_id][collected_face_id[j]][ii][jj]), curr_dL_daa_face_verts[ii][jj]);
			}
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

		int len_oarea_buffer,								    // K
		const torch::Tensor& oarea_buffer_oarea,			    // (B, H, W, K)
		const torch::Tensor& oarea_buffer_tri_id,			    // (B, H, W, K)
		const torch::Tensor& oarea_buffer_tri_cnt,			    // (B, H, W)
		const torch::Tensor& oarea_buffer_doarea_dtri_verts,	// (B, H, W, K, 3, 2)

		// outgoing gradient
		torch::Tensor& dL_dverts,						// (P, 3)
		torch::Tensor& dL_dverts_color,					// (P, 3)
		torch::Tensor& dL_dfaces_opacity,				// (F)
		torch::Tensor& dL_dverts_ndc,					// (B, P, 3)
		torch::Tensor& dL_dfaces_intense,				// (B, F)
		torch::Tensor& dL_daa_face_verts				// (B, F, 3, 2)
	) {
        torch::Tensor aa_face_verts_min = std::get<0>(aa_face_verts.min(2));            // (B, F, 2)
		torch::Tensor aa_face_verts_max = std::get<0>(aa_face_verts.max(2));            // (B, F, 2)

        renderCUDA<3> << <grid, block >> >(
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

            image_ray_o.packed_accessor64<float, 4>(),
            image_ray_d.packed_accessor64<float, 4>(),

            dL_dout_color.packed_accessor64<float, 4>(),
            dL_dout_depth.packed_accessor64<float, 3>(),

            ranges,
            final_Ts,
            final_prev_Ts,
            n_contrib,
            face_list,

            len_oarea_buffer,
            oarea_buffer_oarea.packed_accessor64<float, 4>(),
            oarea_buffer_tri_id.packed_accessor64<int, 4>(),
            oarea_buffer_tri_cnt.packed_accessor64<int, 3>(),
            oarea_buffer_doarea_dtri_verts.packed_accessor64<float, 6>(),

            dL_dverts.packed_accessor64<float, 2>(),
            dL_dverts_color.packed_accessor64<float, 2>(),
            dL_dfaces_opacity.packed_accessor64<float, 1>(),
            dL_dverts_ndc.packed_accessor64<float, 3>(),
            dL_dfaces_intense.packed_accessor64<float, 2>(),
            dL_daa_face_verts.packed_accessor64<float, 4>()
		);
    }
}