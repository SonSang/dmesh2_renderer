import torch as th
from typing import NamedTuple

EPS = 1e-3

class Triangles:
    def __init__(self, p0, p1, p2):
        p0, p1, p2 = order_ccw(p0, p1, p2)                              # [# tri, 2]
        self.verts = th.stack([p0, p1, p2], dim=1)                      # [# tri, 3, 2]
        self.edges = th.stack([p1 - p0, p2 - p1, p0 - p2], dim=1)       # [# tri, 3, 2]
        self.edges_iszero = self.edges.abs() < EPS                      # [# tri, 3, 2]
        self.edges_recip = 1.0 / self.edges                             # [# tri, 3, 2]

        '''
        Equation for triangle edges to find out if a point is inside the triangle:
        '''
        edge_normal_0 = (p1 - p0)[:, [1, 0]]
        edge_normal_0[:, 0] *= -1
        edge_normal_c_0 = (edge_normal_0 * p0).sum(dim=1)               # [# tri]

        edge_normal_1 = (p2 - p1)[:, [1, 0]]
        edge_normal_1[:, 0] *= -1
        edge_normal_c_1 = (edge_normal_1 * p1).sum(dim=1)               # [# tri]

        edge_normal_2 = (p0 - p2)[:, [1, 0]]
        edge_normal_2[:, 0] *= -1
        edge_normal_c_2 = (edge_normal_2 * p2).sum(dim=1)               # [# tri]

        self.edges_normal = th.stack([edge_normal_0, edge_normal_1, edge_normal_2], dim=1)  # [# tri, 3, 2]
        self.edges_normal_c = th.stack([edge_normal_c_0, edge_normal_c_1, edge_normal_c_2], dim=1)  # [# tri, 3]
        
class Pixels:
    def __init__(self, pixmin, pixmax):
        p0 = th.stack([pixmin[:, 0], pixmin[:, 1]], dim=1)              # [# pix, 2]
        p1 = th.stack([pixmax[:, 0], pixmin[:, 1]], dim=1)              # [# pix, 2]
        p2 = th.stack([pixmax[:, 0], pixmax[:, 1]], dim=1)              # [# pix, 2]
        p3 = th.stack([pixmin[:, 0], pixmax[:, 1]], dim=1)              # [# pix, 2]
        self.verts = th.stack([p0, p1, p2, p3], dim=1)                      # [# pix, 4, 2]
        self.edges = th.stack([p1 - p0, p2 - p1, p3 - p2, p0 - p3], dim=1)  # [# pix, 4, 2]

def is_vert_inside_triangle_edge(vert, tri_edge_normal, tri_edge_normal_c):
    '''
    Check if a vertex is inside the triangle edge.
    '''
    vert_normal = (tri_edge_normal * vert).sum()
    return vert_normal - tri_edge_normal_c >= 0

def is_vert_inside_triangle(vert, tri_edges_normal, tri_edges_normal_c):
    '''
    Check if a vertex is inside the triangle.
    '''
    vert = vert.unsqueeze(0)    # [1, 2]
    vert_normal = (tri_edges_normal * vert).sum(dim=1)  # [3]
    return (vert_normal - tri_edges_normal_c).min() >= 0

def is_vert_inside_pixel(vert, pix_verts):
    '''
    Check if a vertex is inside the pixel.
    '''
    xmin = pix_verts[0, 0]
    xmax = pix_verts[1, 0]
    ymin = pix_verts[1, 1]
    ymax = pix_verts[2, 1]
    return xmin <= vert[0] <= xmax and ymin <= vert[1] <= ymax

def _tri_pixel_overlap_area_autograd(tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_verts_is_inside_tri, pix_area):
    polygon = []
    device = tri_verts.device

    pxmin = pix_verts[0, 0]
    pxmax = pix_verts[1, 0]
    pymin = pix_verts[1, 1]
    pymax = pix_verts[2, 1]

    for ti in range(3):
        tri_p0 = tri_verts[ti]
        tri_p1 = tri_verts[(ti + 1) % 3]
        tri_edge = tri_edges[ti]
        is_tri_edge_horizontal = tri_edges_iszero[ti, 1]
        is_tri_edge_vertical = tri_edges_iszero[ti, 0]

        # check if the end points of the edge are inside the pixel
        is_tri_p0_inside = is_vert_inside_pixel(tri_p0, pix_verts)
        is_tri_p1_inside = is_vert_inside_pixel(tri_p1, pix_verts)

        inter_point = []
        inter_t = []
        inter_pedge_idx = []
        for pi in range(4):

            is_pedge_horizontal = ((pi == 0) or (pi == 2))
            is_pedge_vertical = not is_pedge_horizontal
            is_tedge_parallel = (is_tri_edge_horizontal and is_pedge_horizontal) or (is_tri_edge_vertical and is_pedge_vertical)

            if is_pedge_horizontal:
                # pixel edge equation: y = c, so use y axis for intersection
                axis0 = 1
                pmin1, pmax1 = pxmin, pxmax
            else:
                # pixel edge equation: x = c, so use x axis for intersection
                axis0 = 0
                pmin1, pmax1 = pymin, pymax
            axis1 = 1 - axis0

            iaxis0 = pix_verts[pi, axis0]
            t = (iaxis0 - tri_p0[axis0]) * tri_edges_recip[ti, axis0]
            iaxis1 = tri_p0[axis1] + t * tri_edge[axis1]
            
            ipoint = th.zeros((2,), dtype=th.float32, device=device)
            ipoint[axis0] = iaxis0
            ipoint[axis1] = iaxis1

            is_t_valid = ((t >= 0) and (t <= 1) and (iaxis1 >= pmin1) and (iaxis1 <= pmax1) and (not is_tedge_parallel))
            if not is_t_valid:
                continue

            is_ipoint_pixvert = ((iaxis1 == pmin1) or (iaxis1 == pmax1))
            if is_ipoint_pixvert:
                raise ValueError("[pyrasterizer] Error code 00")

            inter_point.append(ipoint)
            inter_t.append(t)
            inter_pedge_idx.append(pi)
                
        num_intersections = len(inter_t)
        if num_intersections > 2:
            raise ValueError("[pyrasterizer] Error code 01")
        
        if num_intersections == 2:
            # sort based on [t]
            if inter_t[0] > inter_t[1]:
                inter_point[0], inter_point[1] = inter_point[1], inter_point[0]
                inter_t[0], inter_t[1] = inter_t[1], inter_t[0]
                inter_pedge_idx[0], inter_pedge_idx[1] = inter_pedge_idx[1], inter_pedge_idx[0]
            
            # add the intersection points to the polygon
            polygon.append(inter_point[0])
            polygon.append(inter_point[1])

            # add pixel vertices to the polygon
            final_pedge_id = inter_pedge_idx[-1]
            start_pvert_id = (final_pedge_id + 1) % 4       # start from this pixel vertex and see if it is inside the triangle
            for pvi in range(4):
                curr_pvert_id = (start_pvert_id + pvi) % 4
                curr_pvert = pix_verts[curr_pvert_id]
                curr_pvert_is_inside = pix_verts_is_inside_tri[curr_pvert_id]
                if curr_pvert_is_inside:
                    polygon.append(curr_pvert)
                    continue
                break

        elif num_intersections == 1:
            # add intersection point to the polygon
            polygon.append(inter_point[0])

            # there is an end point of the triangle edge inside the pixel
            if not is_tri_p0_inside and is_tri_p1_inside:
                polygon.append(tri_p1)
            elif is_tri_p0_inside and not is_tri_p1_inside:
                # skip to remove duplicate intersection point
                # polygon.append(tri_p0)

                # add pixel vertices to the polygon
                final_pedge_id = inter_pedge_idx[-1]
                start_pvert_id = (final_pedge_id + 1) % 4       # start from this pixel vertex and see if it is inside the triangle
                for pvi in range(4):
                    curr_pvert_id = (start_pvert_id + pvi) % 4
                    curr_pvert = pix_verts[curr_pvert_id]
                    curr_pvert_is_inside = pix_verts_is_inside_tri[curr_pvert_id]
                    if curr_pvert_is_inside:
                        polygon.append(curr_pvert)
                        continue
                    break
            else:
                raise ValueError("[pyrasterizer] Error code 02")
            
        else:

            if is_tri_p0_inside and is_tri_p1_inside:
                # add both end points of the edge to the polygon
                # polygon.append(tri_p0)        # skip to remove duplicate intersection point
                polygon.append(tri_p1)
                continue
            elif not is_tri_p0_inside and not is_tri_p1_inside:
                # skip this edge, as it is outside of the pixel and does not intersect with pixel
                continue
            else:
                raise ValueError("[pyrasterizer] Error code 03")

    ### compute area of the polygon
    polyarea = 0.0
    num_subtris = len(polygon) - 2
    for si in range(num_subtris):
        p0 = polygon[0].unsqueeze(0)
        p1 = polygon[si + 1].unsqueeze(0)
        p2 = polygon[si + 2].unsqueeze(0)
        area = tri_area(p0, p1, p2)[0]
        if area < 0:
            raise ValueError("[pyrasterizer] Error code 04")
        polyarea += area

    if polyarea > pix_area:
        raise ValueError("[pyrasterizer] Error code 05")

    return polyarea, polygon

def _tri_pixel_overlap_area_analytic(tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_verts_is_inside_tri, pix_area):
    
    ### gradients
    grad_tri_verts = th.zeros_like(tri_verts)
    
    polygon = []
    polygon_tri_edge_idx = []
    polygon_grad_ip_p0 = []
    polygon_grad_ip_p1 = []

    device = tri_verts.device

    pxmin = pix_verts[0, 0]
    pxmax = pix_verts[1, 0]
    pymin = pix_verts[1, 1]
    pymax = pix_verts[2, 1]

    zero_2b2 = th.zeros(2, 2, device=device, dtype=th.float32)
    eye_2b2 = th.eye(2, device=device, dtype=th.float32)

    for ti in range(3):
        tri_p0 = tri_verts[ti]
        tri_p1 = tri_verts[(ti + 1) % 3]
        tri_edge = tri_edges[ti]
        tri_edge_recip = tri_edges_recip[ti]
        is_tri_edge_horizontal = tri_edges_iszero[ti, 1]
        is_tri_edge_vertical = tri_edges_iszero[ti, 0]

        # check if the end points of the edge are inside the pixel
        is_tri_p0_inside = is_vert_inside_pixel(tri_p0, pix_verts)
        is_tri_p1_inside = is_vert_inside_pixel(tri_p1, pix_verts)

        inter_point = []
        inter_t = []
        inter_pedge_idx = []
        inter_grad_p0 = []
        inter_grad_p1 = []
        for pi in range(4):

            is_pedge_horizontal = ((pi == 0) or (pi == 2))
            is_pedge_vertical = not is_pedge_horizontal
            is_tedge_parallel = (is_tri_edge_horizontal and is_pedge_horizontal) or (is_tri_edge_vertical and is_pedge_vertical)

            if is_pedge_horizontal:
                # pixel edge equation: y = c, so use y axis for intersection
                axis0 = 1
                pmin1, pmax1 = pxmin, pxmax
            else:
                # pixel edge equation: x = c, so use x axis for intersection
                axis0 = 0
                pmin1, pmax1 = pymin, pymax
            axis1 = 1 - axis0

            iaxis0 = pix_verts[pi, axis0]
            t = (iaxis0 - tri_p0[axis0]) * tri_edges_recip[ti, axis0]
            iaxis1 = tri_p0[axis1] + t * tri_edge[axis1]
            
            ipoint = th.zeros((2,), dtype=th.float32, device=device)
            ipoint[axis0] = iaxis0
            ipoint[axis1] = iaxis1

            is_t_valid = ((t >= 0) and (t <= 1) and (iaxis1 >= pmin1) and (iaxis1 <= pmax1) and (not is_tedge_parallel))
            if not is_t_valid:
                continue

            is_ipoint_pixvert = ((iaxis1 == pmin1) or (iaxis1 == pmax1))
            if is_ipoint_pixvert:
                raise ValueError("[pyrasterizer] Error code 00")

            # compute gradient of t w.r.t. tri_p0 and tri_p1
            grad_t_p0 = th.zeros(2, device=device, dtype=th.float32)
            grad_t_p1 = th.zeros(2, device=device, dtype=th.float32)
            grad_t_p0[axis0] = (iaxis0 - tri_p1[axis0]) * tri_edge_recip[axis0] * tri_edge_recip[axis0]
            grad_t_p1[axis0] = (-iaxis0 + tri_p0[axis0]) * tri_edge_recip[axis0] * tri_edge_recip[axis0]
            
            ### compute gradient of ipoint w.r.t. tri_p0 and tri_p1
            grad_ipoint_p0 = ((1.0 - t) * eye_2b2) + (grad_t_p0.unsqueeze(1) @ tri_edge.unsqueeze(0))
            grad_ipoint_p1 = (t * eye_2b2) + (grad_t_p1.unsqueeze(1) @ tri_edge.unsqueeze(0))

            inter_point.append(ipoint)
            inter_t.append(t)
            inter_pedge_idx.append(pi)
            inter_grad_p0.append(grad_ipoint_p0)
            inter_grad_p1.append(grad_ipoint_p1)
                
        num_intersections = len(inter_t)
        if num_intersections > 2:
            raise ValueError("[pyrasterizer] Error code 01")
        
        if num_intersections == 2:
            # sort based on [t]
            if inter_t[0] > inter_t[1]:
                inter_point[0], inter_point[1] = inter_point[1], inter_point[0]
                inter_t[0], inter_t[1] = inter_t[1], inter_t[0]
                inter_pedge_idx[0], inter_pedge_idx[1] = inter_pedge_idx[1], inter_pedge_idx[0]
                inter_grad_p0[0], inter_grad_p0[1] = inter_grad_p0[1], inter_grad_p0[0]
                inter_grad_p1[0], inter_grad_p1[1] = inter_grad_p1[1], inter_grad_p1[0]
            
            # add the intersection points to the polygon
            polygon.append(inter_point[0])
            polygon_tri_edge_idx.append(ti)
            polygon_grad_ip_p0.append(inter_grad_p0[0])
            polygon_grad_ip_p1.append(inter_grad_p1[0])

            polygon.append(inter_point[1])
            polygon_tri_edge_idx.append(ti)
            polygon_grad_ip_p0.append(inter_grad_p0[1])
            polygon_grad_ip_p1.append(inter_grad_p1[1])

            # add pixel vertices to the polygon
            final_pedge_id = inter_pedge_idx[-1]
            start_pvert_id = (final_pedge_id + 1) % 4       # start from this pixel vertex and see if it is inside the triangle
            for pvi in range(4):
                curr_pvert_id = (start_pvert_id + pvi) % 4
                curr_pvert = pix_verts[curr_pvert_id]
                curr_pvert_is_inside = pix_verts_is_inside_tri[curr_pvert_id]
                if curr_pvert_is_inside:
                    polygon.append(curr_pvert)
                    polygon_tri_edge_idx.append(-1)
                    polygon_grad_ip_p0.append(zero_2b2)
                    polygon_grad_ip_p1.append(zero_2b2)
                    continue
                break

        elif num_intersections == 1:
            # add intersection point to the polygon
            polygon.append(inter_point[0])
            polygon_tri_edge_idx.append(ti)
            polygon_grad_ip_p0.append(inter_grad_p0[0])
            polygon_grad_ip_p1.append(inter_grad_p1[0])

            # there is an end point of the triangle edge inside the pixel
            if not is_tri_p0_inside and is_tri_p1_inside:
                polygon.append(tri_p1)
                polygon_tri_edge_idx.append(ti)
                polygon_grad_ip_p0.append(zero_2b2)
                polygon_grad_ip_p1.append(eye_2b2)

            elif is_tri_p0_inside and not is_tri_p1_inside:
                # skip to remove duplicate intersection point
                # polygon.append(tri_p0)

                # add pixel vertices to the polygon
                final_pedge_id = inter_pedge_idx[-1]
                start_pvert_id = (final_pedge_id + 1) % 4       # start from this pixel vertex and see if it is inside the triangle
                for pvi in range(4):
                    curr_pvert_id = (start_pvert_id + pvi) % 4
                    curr_pvert = pix_verts[curr_pvert_id]
                    curr_pvert_is_inside = pix_verts_is_inside_tri[curr_pvert_id]
                    if curr_pvert_is_inside:
                        polygon.append(curr_pvert)
                        polygon_tri_edge_idx.append(-1)
                        polygon_grad_ip_p0.append(zero_2b2)
                        polygon_grad_ip_p1.append(zero_2b2)
                        continue
                    break
            else:
                raise ValueError("[pyrasterizer] Error code 02")
            
        else:

            if is_tri_p0_inside and is_tri_p1_inside:
                # add both end points of the edge to the polygon
                # polygon.append(tri_p0)        # skip to remove duplicate intersection point
                polygon.append(tri_p1)
                polygon_tri_edge_idx.append(ti)
                polygon_grad_ip_p0.append(zero_2b2)
                polygon_grad_ip_p1.append(eye_2b2)
                continue
            elif not is_tri_p0_inside and not is_tri_p1_inside:
                # skip this edge, as it is outside of the pixel and does not intersect with pixel
                continue
            else:
                raise ValueError("[pyrasterizer] Error code 03")

    ### compute area of the polygon
    polyarea = 0.0
    num_subtris = len(polygon) - 2
    for si in range(num_subtris):
        ip0 = polygon[0]
        ip1 = polygon[si + 1]
        ip2 = polygon[si + 2]
        area = tri_area(ip0.unsqueeze(0), ip1.unsqueeze(0), ip2.unsqueeze(0))[0]
        if area < 0:
            raise ValueError("[pyrasterizer] Error code 04")
        polyarea += area

        ### gradient computation
        grad_area_ip0 = 0.5 * th.stack([ip1[1] - ip2[1], -ip1[0] + ip2[0]])
        grad_area_ip1 = 0.5 * th.stack([ip2[1] - ip0[1], -ip2[0] + ip0[0]])
        grad_area_ip2 = 0.5 * th.stack([ip0[1] - ip1[1], -ip0[0] + ip1[0]]) 

        ip0_tri_edge_idx = polygon_tri_edge_idx[0]
        ip1_tri_edge_idx = polygon_tri_edge_idx[si + 1]
        ip2_tri_edge_idx = polygon_tri_edge_idx[si + 2]

        grad_ip0_p0, grad_ip0_p1 = polygon_grad_ip_p0[0], polygon_grad_ip_p1[0]
        grad_ip1_p0, grad_ip1_p1 = polygon_grad_ip_p0[si + 1], polygon_grad_ip_p1[si + 1]
        grad_ip2_p0, grad_ip2_p1 = polygon_grad_ip_p0[si + 2], polygon_grad_ip_p1[si + 2]

        def update_grad_tri_verts(grad_ip_p0, grad_ip_p1, grad_area_ip, ip_tri_edge_idx, t_grad_tri_verts):
            grad_area_p0 = grad_ip_p0 @ grad_area_ip
            grad_area_p1 = grad_ip_p1 @ grad_area_ip
            i0, i1 = ip_tri_edge_idx, (ip_tri_edge_idx + 1) % 3
            t_grad_tri_verts[i0] += grad_area_p0
            t_grad_tri_verts[i1] += grad_area_p1
            return t_grad_tri_verts

        # ip0
        grad_tri_verts = update_grad_tri_verts(grad_ip0_p0, grad_ip0_p1, grad_area_ip0, ip0_tri_edge_idx, grad_tri_verts)
        # ip1
        grad_tri_verts = update_grad_tri_verts(grad_ip1_p0, grad_ip1_p1, grad_area_ip1, ip1_tri_edge_idx, grad_tri_verts)
        # ip2
        grad_tri_verts = update_grad_tri_verts(grad_ip2_p0, grad_ip2_p1, grad_area_ip2, ip2_tri_edge_idx, grad_tri_verts)

    if polyarea > pix_area:
        raise ValueError("[pyrasterizer] Error code 05")

    return polyarea, polygon, grad_tri_verts

class PixelTriangleOverlapAutograd(th.nn.Module):

    @staticmethod
    def forward(tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_verts_is_inside_tri, pix_area):
        return _tri_pixel_overlap_area_autograd(tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_verts_is_inside_tri, pix_area)

class PixelTriangleOverlapAnalytic(th.autograd.Function):

    @staticmethod
    def forward(ctx, tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_verts_is_inside_tri, pix_area):
        ctx.save_for_backward(tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_area)
        ctx.pix_verts_is_inside_tri = pix_verts_is_inside_tri
        try:
            polyarea, polygon, grad_tri_verts = _tri_pixel_overlap_area_analytic(tri_verts, tri_edges, tri_edges_iszero, tri_edges_recip, pix_verts, pix_verts_is_inside_tri, pix_area)
        except ValueError as e:
            print(e)
            polyarea, polygon, grad_tri_verts = 0.0, [], None
        ctx.grad_tri_verts = grad_tri_verts
        
        return polyarea, polygon

    @staticmethod
    def backward(ctx, grad_parea, grad_polygon):
        grad_tri_verts = ctx.grad_tri_verts
        if grad_tri_verts is None:
            return None, None, None, None, None, None, None
        grad_tri_verts = grad_tri_verts * grad_parea
        return grad_tri_verts, None, None, None, None, None, None

def tri_pixel_overlap_area(tris: Triangles, pixs: Pixels, tid, pid, use_autograd):
    '''
    Compute the overlap area of a triangle and a pixel.
    '''
    
    tri_verts = tris.verts[tid]
    tri_edges = tris.edges[tid]
    tri_edges_iszero = tris.edges_iszero[tid]
    tri_edges_recip = tris.edges_recip[tid]
    tri_edges_normal = tris.edges_normal[tid]
    tri_edges_normal_c = tris.edges_normal_c[tid]

    pix_verts = pixs.verts[pid]

    pxmin = pix_verts[0, 0]
    pxmax = pix_verts[1, 0]
    pymin = pix_verts[1, 1]
    pymax = pix_verts[2, 1]    
    pxarea = (pxmax - pxmin) * (pymax - pymin)
    assert pxmin < pxmax, f"pxmin: {pxmin}, pxmax: {pxmax}"
    assert pymin < pymax, f"pymin: {pymin}, pymax: {pymax}"

    # check if the pixel vertices are inside the triangle
    pix_verts_is_inside = [True, True, True, True]
    every_pix_vert_is_inside_triangle = True
    for ti in range(3):
        every_pix_vert_is_outside_triangle_edge = True
        tri_edge_normal, tri_edge_normal_c = tri_edges_normal[ti], tri_edges_normal_c[ti]
        for pvi in range(4):
            pix_vert = pix_verts[pvi]
            pix_vert_is_inside_triangle_edge = is_vert_inside_triangle_edge(pix_vert, tri_edge_normal, tri_edge_normal_c)
            
            every_pix_vert_is_outside_triangle_edge = every_pix_vert_is_outside_triangle_edge and (not pix_vert_is_inside_triangle_edge)
            every_pix_vert_is_inside_triangle = every_pix_vert_is_inside_triangle and pix_vert_is_inside_triangle_edge
            pix_verts_is_inside[pvi] = pix_verts_is_inside[pvi] and pix_vert_is_inside_triangle_edge
        
        if every_pix_vert_is_outside_triangle_edge:
            # if every pixel vertex is outside of a single triangle edge, then the pixel is outside of the triangle
            return 0.0, []
    
    # if every pixel vertex is inside the triangle, return the pixel area
    if every_pix_vert_is_inside_triangle:
        return pxarea, pix_verts

    if use_autograd:
        return PixelTriangleOverlapAutograd.forward(
            tri_verts, 
            tri_edges, 
            tri_edges_iszero, 
            tri_edges_recip,
            pix_verts, 
            pix_verts_is_inside, 
            pxarea
        )
    else:
        return PixelTriangleOverlapAnalytic.apply(
            tri_verts,
            tri_edges,
            tri_edges_iszero,
            tri_edges_recip,
            pix_verts,
            pix_verts_is_inside,
            pxarea
        )

def order_ccw(p0, p1, p2):
    '''
    Order three points in counter-clockwise order.
    '''
    areas = tri_area(p0, p1, p2)
    ### swap where area is negative
    swap_idx = areas < 0
    p1[swap_idx], p2[swap_idx] = p2[swap_idx], p1[swap_idx]
    return p0, p1, p2
    
def tri_area(p0, p1, p2):
    '''
    Compute triangle area. Does not check if points are in ccw order.
    '''
    return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))

class RenderSettings(NamedTuple):
    image_height: int
    image_width: int
    bg : th.Tensor

def render(
    verts: th.Tensor,
    faces: th.Tensor,
    verts_color: th.Tensor,
    faces_opacity: th.Tensor,

    mv_mats: th.Tensor,
    proj_mats: th.Tensor,
    verts_depth: th.Tensor,
    faces_intense: th.Tensor,

    render_settings: RenderSettings,
):
    inv_mv_mats = th.inverse(mv_mats)
    inv_proj_mats = th.inverse(proj_mats)

    img_width = render_settings.image_width
    img_height = render_settings.image_height

    ### preprocess points
    verts_ndc, verts_img = preproces_point(verts, mv_mats, proj_mats, img_width, img_height)

    ### preprocess faces
    face_verts_img, face_depths = preprocess_face(faces, verts_ndc, verts_img, mv_mats, proj_mats, img_width, img_height)

    ### generate rays
    ray_o, ray_d = generate_rays(inv_mv_mats, inv_proj_mats, img_width, img_height)

    ### render

    pass

def preproces_point(verts, mv_mats, proj_mats, width, height):
    ### transform points to ndc
    h_verts = th.cat([verts, th.ones(verts.shape[0], 1)], dim=1)    # [v, 4]
    p_view = th.matmul(mv_mats, h_verts.t())                        # [b, 4, v]
    p_proj = th.matmul(proj_mats, p_view).transpose(1, 2)           # [b, v, 4]

    def clamp_w(w, eps=1e-4):
        w[w == 0.0] = eps
        w[w.abs() < eps] = eps * w[w.abs() < eps].sign()
        return w

    p_w = 1.0 / clamp_w(p_proj[..., [3]])
    p_ndc = p_proj[..., :3] * p_w
    verts_ndc = p_ndc

    ### transform ndc to screen space
    verts_image_x = (verts_ndc[..., 0] + 1.0) * 0.5 * width
    verts_image_y = (verts_ndc[..., 1] + 1.0) * 0.5 * height
    verts_image = th.stack([verts_image_x, verts_image_y], dim=-1)

    return verts_ndc, verts_image

def preprocess_face(faces, verts_ndc, verts_img, mv, proj, width, height):
    num_batch = mv.shape[0]
    num_faces = faces.shape[0]
    face_verts_image = th.zeros((num_batch, num_faces, 3, 2), dtype=th.float32, device=faces.device)
    face_depths = th.zeros((num_batch, num_faces), dtype=th.float32, device=faces.device)

    for ti in range(3):
        face_vert_id = faces[:, ti]     # [F]
        face_vert_ndc = verts_ndc[:, face_vert_id]    # [B, F, 3]
        z = face_vert_ndc[:, :, 2]      # [B, F]

        face_depths += z
        face_verts_image[:, :, ti] = verts_img[:, face_vert_id, :2]
    face_depths = ((face_depths / 3.0) + 1.0) * 0.5

    return face_verts_image, face_depths

def generate_rays(inv_mv, inv_proj, width, height):
    
    num_batch = inv_mv.shape[0]

    ray_o = th.zeros((num_batch, height, width, 3), dtype=th.float32, device=inv_mv.device)
    ray_d = th.zeros((num_batch, height, width, 3), dtype=th.float32, device=inv_mv.device)

    ray_o[:, :, :, 0] = inv_mv[:, 3, 0][:, None, None]
    ray_o[:, :, :, 1] = inv_mv[:, 3, 1][:, None, None]
    ray_o[:, :, :, 2] = inv_mv[:, 3, 2][:, None, None]

    pixel_id = th.arange(width * height, device=inv_mv.device)
    pixel_id = pixel_id.view(1, height, width)
    pixel_id = pixel_id.expand(num_batch, height, width)        # [B, H, W]

    pixel_x = pixel_id % width
    pixel_y = pixel_id // width

    pixf = th.stack([pixel_x + 0.5, pixel_y + 0.5], dim=-1)     # [B, H, W, 2]

    ### transform pixel coords to world coords
    # first, bring it to NDC
    pix_ndc_x = pixf[..., 0] / width * 2.0 - 1.0
    pix_ndc_y = pixf[..., 1] / height * 2.0 - 1.0
    pix_ndc = th.stack([pix_ndc_x, pix_ndc_y], dim=-1)                      # [B, H, W, 2]
    pix_ndc = th.cat([pix_ndc, th.ones_like(pix_ndc[..., :1])], dim=-1)     # [B, H, W, 3]
    pix_ndc = th.cat([pix_ndc, th.ones_like(pix_ndc[..., :1])], dim=-1)     # [B, H, W, 4]
    pix_ndc = pix_ndc.view(num_batch, -1, 4)                                # [B, HW, 4]

    # then, bring it to world space
    pix_view = th.matmul(inv_proj, pix_ndc.transpose(1, 2))                 # [B, 4, HW]
    pix_world = th.matmul(inv_mv, pix_view)                                 # [B, 4, HW]
    pix_world = pix_world.transpose(1, 2)                                   # [B, HW, 4]
    pix_world = pix_world[..., :3]                                          # [B, HW, 3]
    ray_target = pix_world.view(num_batch, height, width, 3)                 # [B, H, W, 3]

    ray_d = ray_target - ray_o
    ray_len = th.sqrt(th.sum(ray_d ** 2, dim=-1, keepdim=True)) + 1e-6
    ray_d = ray_d / ray_len
    
    return ray_o, ray_d