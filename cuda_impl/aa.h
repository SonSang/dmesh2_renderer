#ifndef CUDA_AA_H_INCLUDED
#define CUDA_AA_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

#define MAX_NUM_POLYGONS    10

namespace AA
{
    __forceinline__ __device__ bool _is_vert_inside_triangle_edge(
        const float* vert,
        const float* tri_edge_normal,
        const float tri_edge_normal_c
    ) {
        return (vert[0] * tri_edge_normal[0]) + (vert[1] * tri_edge_normal[1]) - tri_edge_normal_c >= 0;
    }

    __forceinline__ __device__ bool _is_vert_inside_pixel(
        const float* vert,
        const float pxmin,
        const float pxmax,
        const float pymin,
        const float pymax
    ) {
        return (vert[0] >= pxmin) && (vert[0] <= pxmax) && (vert[1] >= pymin) && (vert[1] <= pymax);
    }

    __forceinline__ __device__ bool _add_polygon(
        float (*polygon)[2],
        int* polygon_tri_edge_idx,
        float (*polygon_grad_ip_p0)[4],
        float (*polygon_grad_ip_p1)[4],
        int* num_polygons,

        const float* n_polygon,
        const int n_polygon_tri_edge_idx,
        const float* n_polygon_grad_ip_p0,
        const float* n_polygon_grad_ip_p1
    ) {
        if (*num_polygons >= MAX_NUM_POLYGONS) {
            printf("[AA] Error: Polygon buffer overflow\n");
            return false;
        }
        polygon[*num_polygons][0] = n_polygon[0];
        polygon[*num_polygons][1] = n_polygon[1];
        
        polygon_grad_ip_p0[*num_polygons][0] = n_polygon_grad_ip_p0[0];
        polygon_grad_ip_p0[*num_polygons][1] = n_polygon_grad_ip_p0[1];
        polygon_grad_ip_p0[*num_polygons][2] = n_polygon_grad_ip_p0[2];
        polygon_grad_ip_p0[*num_polygons][3] = n_polygon_grad_ip_p0[3];
        polygon_grad_ip_p1[*num_polygons][0] = n_polygon_grad_ip_p1[0];
        polygon_grad_ip_p1[*num_polygons][1] = n_polygon_grad_ip_p1[1];
        polygon_grad_ip_p1[*num_polygons][2] = n_polygon_grad_ip_p1[2];
        polygon_grad_ip_p1[*num_polygons][3] = n_polygon_grad_ip_p1[3];

        polygon_tri_edge_idx[*num_polygons] = n_polygon_tri_edge_idx;
        
        (*num_polygons)++;
        return true;
    }

    __forceinline__ __device__ void _update_grad_tri_verts(
        const float* grad_ip_p0,            // [4]
        const float* grad_ip_p1,            // [4]
        const float* grad_area_ip,          // [2]
        const int ip_tri_edge_idx,
        float (*grad_tri_verts)[2]               // [3, 2]
    ) {
        float grad_area_p0[2];
        float grad_area_p1[2];
        grad_area_p0[0] = grad_ip_p0[0] * grad_area_ip[0] + grad_ip_p0[1] * grad_area_ip[1];
        grad_area_p0[1] = grad_ip_p0[2] * grad_area_ip[0] + grad_ip_p0[3] * grad_area_ip[1];
        grad_area_p1[0] = grad_ip_p1[0] * grad_area_ip[0] + grad_ip_p1[1] * grad_area_ip[1];
        grad_area_p1[1] = grad_ip_p1[2] * grad_area_ip[0] + grad_ip_p1[3] * grad_area_ip[1];
        int i0 = ip_tri_edge_idx;
        int i1 = (ip_tri_edge_idx + 1) % 3;
        grad_tri_verts[i0][0] += grad_area_p0[0];
        grad_tri_verts[i0][1] += grad_area_p0[1];
        grad_tri_verts[i1][0] += grad_area_p1[0];
        grad_tri_verts[i1][1] += grad_area_p1[1];
    }

    __forceinline__ __device__ float _tri_area(
        const float* p0,
        const float* p1,
        const float* p2
    ) {
        return 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]));
    }

    __forceinline__ __device__ bool _is_pix_outside_tri_bb(
        const float txmin, float txmax, float tymin, float tymax,
        const float pxmin, float pxmax, float pymin, float pymax
    ) {
        return (pxmax < txmin) || (pxmin > txmax) || (pymax < tymin) || (pymin > tymax);
    }

    __forceinline__ __device__ bool _is_pix_outside_tri(
        const int batch_id,
        const int tri_id,
        torch::PackedTensorAccessor64<float, 4> tri_edges_normal,       // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 3> tri_edges_normal_c,     // [B, T, 3]
        float (*pix_verts)[2],
        bool* pix_verts_is_inside
    ) {
        float f_tri_edges_normal[3][2] = {
            { tri_edges_normal[batch_id][tri_id][0][0], tri_edges_normal[batch_id][tri_id][0][1] },
            { tri_edges_normal[batch_id][tri_id][1][0], tri_edges_normal[batch_id][tri_id][1][1] },
            { tri_edges_normal[batch_id][tri_id][2][0], tri_edges_normal[batch_id][tri_id][2][1] }
        };
        float f_tri_edges_normal_c[3] = {
            tri_edges_normal_c[batch_id][tri_id][0],
            tri_edges_normal_c[batch_id][tri_id][1],
            tri_edges_normal_c[batch_id][tri_id][2]
        };

        for (int pvi = 0; pvi < 4; pvi++) {
            pix_verts_is_inside[pvi] = true;
        }

        for (int ti = 0; ti < 3; ti++) {
            bool every_pix_vert_is_outside_triangle_edge = true;
            const float* tri_edge_normal = f_tri_edges_normal[ti];
            const float tri_edge_normal_c = f_tri_edges_normal_c[ti];
            
            for (int pvi = 0; pvi < 4; pvi++) {
                float* pix_vert = pix_verts[pvi];
                bool pix_vert_is_inside_triangle_edge = _is_vert_inside_triangle_edge(pix_vert, tri_edge_normal, tri_edge_normal_c);
                
                // if any one of the pixel vertices is inside the triangle edge, we cannot say that the pixel is outside of the triangle
                every_pix_vert_is_outside_triangle_edge = every_pix_vert_is_outside_triangle_edge && (!pix_vert_is_inside_triangle_edge);

                // check if the pixel vertex is inside the every triangle edge
                pix_verts_is_inside[pvi] = pix_verts_is_inside[pvi] && pix_vert_is_inside_triangle_edge;
            }

            // if every pixel vertex is outside of a single triangle edge, then the pixel is outside of the triangle
            if (every_pix_vert_is_outside_triangle_edge) {
                return true;
            }
        }

        return false;
    }

    __forceinline__ __device__ int _sub_tri_pix_overlap_area(
        // tri info
        torch::PackedTensorAccessor64<float, 4> tri_verts,          // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 4> tri_edges,          // [B, T, 3, 2]
        torch::PackedTensorAccessor64<bool, 4> tri_edges_iszero,    // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 4> tri_edges_recip,    // [B, T, 3, 2]
        const int batch_id,
        const int tri_id,
        
        // pix info
        const float pxmin,
        const float pxmax,
        const float pymin,
        const float pymax,
        const float pix_area,
        const float (*pix_verts)[2],    // [4, 2]
        const bool* pix_verts_is_inside,

        // output
        float* area,
        float (*grad_tri_verts)[2]      // [3, 2]
    ) {
        // polygon info
        float polygon[MAX_NUM_POLYGONS][2];
        int polygon_tri_edge_idx[MAX_NUM_POLYGONS];
        float polygon_grad_ip_p0[MAX_NUM_POLYGONS][4];     // 2x2 matrix
        float polygon_grad_ip_p1[MAX_NUM_POLYGONS][4];
        int num_polygons = 0;
        
        // default mats
        float zero2[4] = { 0, 0, 0, 0 };
        float eye2[4] = { 1, 0, 0, 1 };

        float f_tri_verts[3][2] = {
            { tri_verts[batch_id][tri_id][0][0], tri_verts[batch_id][tri_id][0][1] },
            { tri_verts[batch_id][tri_id][1][0], tri_verts[batch_id][tri_id][1][1] },
            { tri_verts[batch_id][tri_id][2][0], tri_verts[batch_id][tri_id][2][1] }
        };
        float f_tri_edges[3][2] = {
            { tri_edges[batch_id][tri_id][0][0], tri_edges[batch_id][tri_id][0][1] },
            { tri_edges[batch_id][tri_id][1][0], tri_edges[batch_id][tri_id][1][1] },
            { tri_edges[batch_id][tri_id][2][0], tri_edges[batch_id][tri_id][2][1] }
        };
        bool b_tri_edges_iszero[3][2] = {
            { tri_edges_iszero[batch_id][tri_id][0][0], tri_edges_iszero[batch_id][tri_id][0][1] },
            { tri_edges_iszero[batch_id][tri_id][1][0], tri_edges_iszero[batch_id][tri_id][1][1] },
            { tri_edges_iszero[batch_id][tri_id][2][0], tri_edges_iszero[batch_id][tri_id][2][1] }
        };
        float f_tri_edges_recip[3][2] = {
            { tri_edges_recip[batch_id][tri_id][0][0], tri_edges_recip[batch_id][tri_id][0][1] },
            { tri_edges_recip[batch_id][tri_id][1][0], tri_edges_recip[batch_id][tri_id][1][1] },
            { tri_edges_recip[batch_id][tri_id][2][0], tri_edges_recip[batch_id][tri_id][2][1] }
        };
        
        // iterate through the edges of the triangle
        for (int ti = 0; ti < 3; ti++) {
            int tri_p0_idx = ti;
            int tri_p1_idx = (ti + 1) % 3;
            
            const float* tri_p0 = f_tri_verts[tri_p0_idx];
            const float* tri_p1 = f_tri_verts[tri_p1_idx];
            const float* tri_edge = f_tri_edges[ti];
            const bool* tri_edge_iszero = b_tri_edges_iszero[ti];
            const float* tri_edge_recip = f_tri_edges_recip[ti];

            bool is_tri_edge_horizontal = tri_edge_iszero[1];
            bool is_tri_edge_vertical = tri_edge_iszero[0];

            // check if the end points of the edge are inside the pixel
            bool is_tri_p0_inside = _is_vert_inside_pixel(tri_p0, pxmin, pxmax, pymin, pymax);
            bool is_tri_p1_inside = _is_vert_inside_pixel(tri_p1, pxmin, pxmax, pymin, pymax);

            float inter_point[4][2];
            float inter_t[4];
            int inter_pedge_idx[4];
            float inter_grad_p0[4][4];
            float inter_grad_p1[4][4];
            int num_intersections = 0;

            for (int pi = 0; pi < 4; pi++) {
                bool is_pedge_horizontal = ((pi == 0) || (pi == 2));
                bool is_pedge_vertical = !is_pedge_horizontal;
                bool is_tedge_parallel = (is_tri_edge_horizontal && is_pedge_horizontal) || (is_tri_edge_vertical && is_pedge_vertical);

                int axis0 = -1;
                float pmin1, pmax1;
                if (is_pedge_horizontal) {
                    // pixel edge equation: y = c, so use y axis for intersection
                    axis0 = 1;
                    pmin1 = pxmin; 
                    pmax1 = pxmax;
                } else {
                    // pixel edge equation: x = c, so use x axis for intersection
                    axis0 = 0;
                    pmin1 = pymin; 
                    pmax1 = pymax;
                }
                int axis1 = 1 - axis0;

                float iaxis0 = pix_verts[pi][axis0];
                float t = (iaxis0 - tri_p0[axis0]) * tri_edge_recip[axis0];
                float iaxis1 = tri_p0[axis1] + t * tri_edge[axis1];

                float ipoint[2];
                ipoint[axis0] = iaxis0;
                ipoint[axis1] = iaxis1;

                bool is_t_valid = ((t >= 0) && (t <= 1) && (iaxis1 >= pmin1) && (iaxis1 <= pmax1) && (!is_tedge_parallel));
                if (!is_t_valid) {
                    continue;
                }

                bool is_ipoint_pixvert = ((iaxis1 == pmin1) || (iaxis1 == pmax1));
                if (is_ipoint_pixvert) {
                    return 1;       // error code 00
                }

                // intersection point
                inter_point[num_intersections][axis0] = iaxis0;
                inter_point[num_intersections][axis1] = iaxis1;

                // intersection t
                inter_t[num_intersections] = t;

                // intersection pixel edge index
                inter_pedge_idx[num_intersections] = pi;

                // intersection point gradient w.r.t. tri_p0 and tri_p1
                // compute gradient of t w.r.t. tri_p0 and tri_p1
                float grad_t_p0[2] = { 0, 0 };
                float grad_t_p1[2] = { 0, 0 };
                grad_t_p0[axis0] = (iaxis0 - tri_p1[axis0]) * tri_edge_recip[axis0] * tri_edge_recip[axis0];
                grad_t_p1[axis0] = (-iaxis0 + tri_p0[axis0]) * tri_edge_recip[axis0] * tri_edge_recip[axis0];

                // compute gradient of ipoint w.r.t. tri_p0 and tri_p1
                inter_grad_p0[num_intersections][0] = (1.0 - t) + (grad_t_p0[0] * tri_edge[0]);
                inter_grad_p0[num_intersections][1] = grad_t_p0[0] * tri_edge[1];
                inter_grad_p0[num_intersections][2] = grad_t_p0[1] * tri_edge[0];
                inter_grad_p0[num_intersections][3] = (1.0 - t) + (grad_t_p0[1] * tri_edge[1]);

                inter_grad_p1[num_intersections][0] = t + (grad_t_p1[0] * tri_edge[0]);
                inter_grad_p1[num_intersections][1] = grad_t_p1[0] * tri_edge[1];
                inter_grad_p1[num_intersections][2] = grad_t_p1[1] * tri_edge[0];
                inter_grad_p1[num_intersections][3] = t + (grad_t_p1[1] * tri_edge[1]);
                
                num_intersections++;
            }

            // if number of intersections is more than 2, error
            if (num_intersections > 2) {
                return 2;       // error code 01
            }

            bool add_polygon_succeed = true;
            if (num_intersections > 0) {
                int final_pedge_id = -1;
                if (num_intersections == 2) {
                    // sort based on [t]
                    int si[2] = { 0, 1 };
                    if (inter_t[0] > inter_t[1]) {
                        si[0] = 1;
                        si[1] = 0;
                    }

                    // add the intersection points to the polygon
                    add_polygon_succeed &= _add_polygon(
                        polygon, polygon_tri_edge_idx, polygon_grad_ip_p0, polygon_grad_ip_p1, &num_polygons,
                        inter_point[si[0]], ti, inter_grad_p0[si[0]], inter_grad_p1[si[0]]
                    );
                    add_polygon_succeed &= _add_polygon(
                        polygon, polygon_tri_edge_idx, polygon_grad_ip_p0, polygon_grad_ip_p1, &num_polygons,
                        inter_point[si[1]], ti, inter_grad_p0[si[1]], inter_grad_p1[si[1]]
                    );
                    if (!add_polygon_succeed) {
                        return 5;       // error code 05
                    }

                    final_pedge_id = inter_pedge_idx[si[1]];
                }
                else {
                    // add intersection point to the polygon
                    add_polygon_succeed &= _add_polygon(
                        polygon, polygon_tri_edge_idx, polygon_grad_ip_p0, polygon_grad_ip_p1, &num_polygons,
                        inter_point[0], ti, inter_grad_p0[0], inter_grad_p1[0]
                    );
                    if (!add_polygon_succeed) {
                        return 5;       // error code 05
                    }

                    // there is an end point of the triangle edge inside the pixel
                    if (!is_tri_p0_inside && is_tri_p1_inside) {
                        add_polygon_succeed &= _add_polygon(
                            polygon, polygon_tri_edge_idx, polygon_grad_ip_p0, polygon_grad_ip_p1, &num_polygons,
                            tri_p1, ti, zero2, eye2
                        );
                        if (!add_polygon_succeed) {
                            return 5;       // error code 05
                        }
                    }
                    else if (is_tri_p0_inside && !is_tri_p1_inside) {
                        // skip polygon addition to remove duplicate intersection point
                        final_pedge_id = inter_pedge_idx[0];
                    }
                    else {
                        return 3;       // error code 02
                    }
                }

                if (final_pedge_id != -1) {
                    // add pixel vertices to the polygon
                    int start_pvert_id = (final_pedge_id + 1) % 4;       // start from this pixel vertex and see if it is inside the triangle
                    for (int pvi = 0; pvi < 4; pvi++) {
                        int curr_pvert_id = (start_pvert_id + pvi) % 4;
                        const float* curr_pvert = pix_verts[curr_pvert_id];
                        bool curr_pvert_is_inside = pix_verts_is_inside[curr_pvert_id];
                        if (curr_pvert_is_inside) {
                            add_polygon_succeed &= _add_polygon(
                                polygon, polygon_tri_edge_idx, polygon_grad_ip_p0, polygon_grad_ip_p1, &num_polygons,
                                curr_pvert, -1, zero2, zero2
                            );
                            if (!add_polygon_succeed) {
                                return 5;       // error code 05
                            }
                        }
                        else {
                            break;
                        }
                    }
                }
            }
            else {
                if (is_tri_p0_inside && is_tri_p1_inside) {
                    // add both end points of the edge to the polygon
                    // skip the first one to remove duplicate intersection point
                    add_polygon_succeed &= _add_polygon(
                        polygon, polygon_tri_edge_idx, polygon_grad_ip_p0, polygon_grad_ip_p1, &num_polygons,
                        tri_p1, ti, zero2, eye2
                    );
                    if (!add_polygon_succeed) {
                        return 5;       // error code 05
                    }
                }
                else if (!is_tri_p0_inside && !is_tri_p1_inside) {
                    // skip this edge, as it is outside of the pixel and does not intersect with pixel
                    continue;
                }
                else {
                    return 4;       // error code 03
                }
            }
        }

        // compute area of the polygon
        int num_subtris = num_polygons - 2;
        for (int si = 0; si < num_subtris; si++) {
            const float* ip0 = polygon[0];
            const float* ip1 = polygon[si + 1];
            const float* ip2 = polygon[si + 2];
            float s_area = _tri_area(ip0, ip1, ip2);
            if (s_area < 0) {
                return 5;       // error code 04
            }
            *area += s_area;

            // gradient computation
            float grad_area_ip0[2] = { 0.5f * (ip1[1] - ip2[1]), 0.5f * (-ip1[0] + ip2[0]) };
            float grad_area_ip1[2] = { 0.5f * (ip2[1] - ip0[1]), 0.5f * (-ip2[0] + ip0[0]) };
            float grad_area_ip2[2] = { 0.5f * (ip0[1] - ip1[1]), 0.5f * (-ip0[0] + ip1[0]) };

            int ip0_tri_edge_idx = polygon_tri_edge_idx[0];
            int ip1_tri_edge_idx = polygon_tri_edge_idx[si + 1];
            int ip2_tri_edge_idx = polygon_tri_edge_idx[si + 2];

            const float* grad_ip0_p0 = polygon_grad_ip_p0[0];
            const float* grad_ip0_p1 = polygon_grad_ip_p1[0];
            const float* grad_ip1_p0 = polygon_grad_ip_p0[si + 1];
            const float* grad_ip1_p1 = polygon_grad_ip_p1[si + 1];
            const float* grad_ip2_p0 = polygon_grad_ip_p0[si + 2];
            const float* grad_ip2_p1 = polygon_grad_ip_p1[si + 2];

            _update_grad_tri_verts(grad_ip0_p0, grad_ip0_p1, grad_area_ip0, ip0_tri_edge_idx, grad_tri_verts);
            _update_grad_tri_verts(grad_ip1_p0, grad_ip1_p1, grad_area_ip1, ip1_tri_edge_idx, grad_tri_verts);
            _update_grad_tri_verts(grad_ip2_p0, grad_ip2_p1, grad_area_ip2, ip2_tri_edge_idx, grad_tri_verts);
        }

        if (*area > pix_area) {
            return 6;       // error code 05
        }

        return 0;       // success
    }

	/*
	Find overlapping area between a triangle and a pixel.
	*/
	inline __device__ int tri_pix_overlap_area(
		// tri info
        torch::PackedTensorAccessor64<float, 4> tri_verts,          // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 4> tri_edges,          // [B, T, 3, 2]
        torch::PackedTensorAccessor64<bool, 4> tri_edges_iszero,    // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 4> tri_edges_recip,    // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 4> tri_edges_normal,   // [B, T, 3, 2]
        torch::PackedTensorAccessor64<float, 3> tri_edges_normal_c, // [B, T, 3]
        const int batch_id,
        const int tri_id,

        const float txmin,
        const float txmax,
        const float tymin,
        const float tymax,

        // pix info
        const float pxmin,
        const float pxmax,
        const float pymin,
        const float pymax,
        const float pix_area,

        // output
        float* area,
        float (*grad_tri_verts)[2]   // [3, 2]
	) {
        // Note: Assume [area] and [grad_tri_verts] are initialized to 0.0f

        // 1. If the pixel is outside of the triangle bounding box, return 0.
        if (_is_pix_outside_tri_bb(txmin, txmax, tymin, tymax, pxmin, pxmax, pymin, pymax)) {
            return 0;
        }

        // 2. Check more precisely if the pixel is outside of the triangle.
        float pix_verts[4][2] = {
            { pxmin, pymin },
            { pxmax, pymin },
            { pxmax, pymax },
            { pxmin, pymax }
        };
        bool pix_verts_is_inside[4];
        if (_is_pix_outside_tri(batch_id, tri_id, tri_edges_normal, tri_edges_normal_c, pix_verts, pix_verts_is_inside)) {
            return 0;
        }

        // 3. If the pixel is completely inside the triangle, return the pixel area.
        if (pix_verts_is_inside[0] && pix_verts_is_inside[1] && pix_verts_is_inside[2] && pix_verts_is_inside[3]) {
            *area = pix_area;
            return 0;
        }

        // 4. Compute the overlap area between the pixel and the triangle.
        return _sub_tri_pix_overlap_area(
            tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, batch_id, tri_id,
            pxmin, pxmax, pymin, pymax, pix_area, pix_verts, pix_verts_is_inside,
            area, grad_tri_verts
        );
    }
}


#endif