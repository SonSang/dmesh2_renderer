from typing import List
import torch as th
from dmesh2_renderer.pyrenderer import Triangles
from . import _C

'''
===============================================================================
3D Renderer
===============================================================================
'''
class RenderFunction(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,

        background,                     # (3),              no grad

        ### patch info
        patch_min,                      # (B, 2),           no grad
        patch_width,
        patch_height,

        ### global info
        verts,                          # (P, 3),           grad
        faces,                          # (F, 3),           no grad
        verts_color,                    # (P, 3),           grad
        faces_opacity,                  # (F,),             grad

        ### local info (per batch)
        verts_ndc,                      # (B, P, 3)         grad (only for last channel, depth)
        verts_image,                    # (B, P, 2)         no grad
        faces_intense,                  # (B, F)            grad

        ### for aa
        aa_temperature,
        aa_face_verts,					# (B, F, 3, 2)      grad
        aa_face_edges,					# (B, F, 3, 2)      no grad
        aa_face_edges_iszero,			# (B, F, 3, 2)      no grad
        aa_face_edges_recip,			# (B, F, 3, 2)      no grad
        aa_face_edges_normal,			# (B, F, 3, 2)      no grad
        aa_face_edges_normal_c,		    # (B, F, 3)         no grad
        len_oarea_buffer,               # int

        ### ray
        image_ray_o,					# (B, H, W, 3)      no grad
        image_ray_d,					# (B, H, W, 3)      no grad
    ):
        args = (
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
            len_oarea_buffer,

            image_ray_o,
            image_ray_d,
        )

        try:
            num_rendered, color, depth, orea_buffer_oarea, oarea_buffer_tri_id, oarea_buffer_tri_cnt, oarea_buffer_doarea_dtri_verts, face_buffer, binning_buffer, image_buffer = _C.render_forward_cuda(*args)
        except Exception as ex:
            print("\nAn error occured in renderer forward.")
            print(ex)
            raise ex
        
        # Keep relevant tensors for backward
        ctx.save_for_backward(
            background,
            patch_min,
            verts,
            faces,
            verts_color,
            faces_opacity,
            verts_ndc,
            verts_image,
            faces_intense,
            aa_face_verts,
            aa_face_edges,
            aa_face_edges_iszero,
            aa_face_edges_recip,
            aa_face_edges_normal,
            aa_face_edges_normal_c,
            image_ray_o,
            image_ray_d,
            orea_buffer_oarea,
            oarea_buffer_tri_id,
            oarea_buffer_tri_cnt,
            oarea_buffer_doarea_dtri_verts,
            face_buffer,
            binning_buffer,
            image_buffer
        )
        ctx.num_rendered = num_rendered
        ctx.patch_width = patch_width
        ctx.patch_height = patch_height
        ctx.aa_temperature = aa_temperature
        ctx.len_oarea_buffer = len_oarea_buffer

        return color, depth
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth):
        background, patch_min, verts, faces, verts_color, faces_opacity, verts_ndc, verts_image, faces_intense, \
            aa_face_verts, aa_face_edges, aa_face_edges_iszero, aa_face_edges_recip, aa_face_edges_normal, aa_face_edges_normal_c, \
                image_ray_o, image_ray_d, oarea_buffer_oarea, oarea_buffer_tri_id, oarea_buffer_tri_cnt, oarea_buffer_doarea_dtri_verts, \
                    face_buffer, binning_buffer, image_buffer = ctx.saved_tensors
        num_rendered, patch_width, patch_height, aa_temperature, len_oarea_buffer = \
            ctx.num_rendered, ctx.patch_width, ctx.patch_height, ctx.aa_temperature, ctx.len_oarea_buffer

        args = (
            num_rendered,
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
            len_oarea_buffer,

            image_ray_o,
            image_ray_d,

            grad_out_color,
            grad_out_depth,

            face_buffer,
            binning_buffer,
            image_buffer,
            oarea_buffer_oarea,
            oarea_buffer_tri_id,
            oarea_buffer_tri_cnt,
            oarea_buffer_doarea_dtri_verts
        )

        try:
            grad_verts, grad_verts_color, grad_faces_opacity, grad_verts_ndc, grad_faces_intense, grad_aa_face_verts = _C.render_backward_cuda(*args)
        except Exception as ex:
            print("\nAn error occured in renderer forward.")
            print(ex)
            raise ex

        return None, None, None, None, grad_verts, None, grad_verts_color, grad_faces_opacity, grad_verts_ndc, \
                None, grad_faces_intense, None, grad_aa_face_verts, None, None, None, None, None, None, None, None
    
class Renderer(th.nn.Module):

    def __init__(self, mv, proj, width, height, device, aa_grad_buffer_size=20):
        super(Renderer, self).__init__()
        self.mv = mv                        # [B, 4, 4]
        self.proj = proj                    # [B, 4, 4]
        self.width = width
        self.height = height
        self.device = device

        self.num_batch = self.mv.shape[0]

        # initialize rays for each pixel
        self.ray_o = None
        self.ray_d = None
        self._init_rays()

        self.aa_grad_buffer_size = aa_grad_buffer_size

    def _init_rays(self):

        # get the inverse mv and proj matrices
        inv_mv = th.inverse(self.mv)                # [B, 4, 4]
        inv_proj = th.inverse(self.proj)            # [B, 4, 4]

        # ray origin is the camera position in world space
        self.ray_o = th.zeros((self.num_batch, self.height, self.width, 3), device=self.device)
        self.ray_o[:, :, :, 0] = inv_mv[:, 0, 3].reshape(-1, 1, 1)            # [B, H, W]
        self.ray_o[:, :, :, 1] = inv_mv[:, 1, 3].reshape(-1, 1, 1)
        self.ray_o[:, :, :, 2] = inv_mv[:, 2, 3].reshape(-1, 1, 1)

        # find out (floating point) pixel coords that we will cast a ray through
        pixel_x = th.arange(self.width, device=self.device).float()
        pixel_y = th.arange(self.height, device=self.device).float()

        pixf = th.stack(th.meshgrid([pixel_x, pixel_y], indexing='ij'), dim=-1).float() + 0.5           # [W, H, 2]
        pixf = pixf.transpose(0, 1)                                                                     # [H, W, 2]
        pixf = pixf.unsqueeze(0).expand(self.num_batch, -1, -1, -1)                 # [B, H, W, 2]

        # transform pixel coords to world coords
        # first, bring it to NDC
        pix_ndc = th.zeros_like(pixf)
        pix_ndc[:, :, :, 0] = (pixf[:, :, :, 0] / self.width * 2) - 1
        pix_ndc[:, :, :, 1] = (pixf[:, :, :, 1] / self.height * 2) - 1

        # then, bring it to world space
        pix_ndc_h = th.cat((pix_ndc, -th.ones_like(pix_ndc[:, :, :, 0:1])), dim=-1)                         # [B, H, W, 3]
        pix_ndc_h = th.cat((pix_ndc_h, th.ones_like(pix_ndc_h[:, :, :, 0:1])), dim=-1)                      # [B, H, W, 4]
        pix_ndc_h = pix_ndc_h.unsqueeze(-2)                                                                 # [B, H, W, 1, 4]
        pix_view = th.matmul(pix_ndc_h, inv_proj.transpose(1, 2).unsqueeze(1).unsqueeze(1))                 # [B, H, W, 1, 4]
        pix_world = th.matmul(pix_view, inv_mv.transpose(1, 2).unsqueeze(1).unsqueeze(1))                   # [B, H, W, 1, 4]
        pix_world = pix_world[..., :3]                                                                      # [B, H, W, 1, 3]
        pix_world = pix_world.squeeze(-2)                                                                   # [B, H, W, 3]
        ray_target = pix_world

        # ray direction is the normalized vector from camera position to pixel position
        ray_d = ray_target - self.ray_o
        ray_d_len = th.norm(ray_d, dim=-1, keepdim=True) + 1e-6
        self.ray_d = ray_d / ray_d_len

    def compute_verts_ndc_image(self, verts, mv, proj):
        '''
        Compute ndc and image coordinates for each vertices using [mv, proj].

        @ verts: [# point, 3]
        @ mv: [# batch, 4, 4]
        @ proj: [# batch, 4, 4]
        '''

        verts_hom = th.cat((verts, th.ones_like(verts[:, [0]])), dim=-1)            # [V, 4]
        verts_view = th.matmul(verts_hom, mv.transpose(1, 2))                       # [B, V, 4]
        verts_proj = th.matmul(verts_view, proj.transpose(1, 2))                    # [B, V, 4]
        verts_proj_w = verts_proj[..., [3]]                                         # [B, V, 1]

        # clamp w;
        verts_proj_w[th.logical_and(verts_proj_w >= 0.0, verts_proj_w < 1e-4)] = 1e-4
        verts_proj_w[th.logical_and(verts_proj_w < 0.0, verts_proj_w > -1e-4)] = -1e-4

        verts_ndc = verts_proj[..., :3] / verts_proj_w                              # [B, V, 3]
        verts_image = (verts_ndc[..., :2] + 1) * 0.5                                # [B, V, 2]
        verts_image[:, :, 0] = verts_image[:, :, 0] * self.width
        verts_image[:, :, 1] = verts_image[:, :, 1] * self.height
        
        return verts_ndc, verts_image

    def select_rays(self, batch_mvp_idx, batch_patch_min, patch_width, patch_height):

        device = self.device
        
        entire_ray_o = self.ray_o[batch_mvp_idx]            # [B, H, W, 3]
        entire_ray_d = self.ray_d[batch_mvp_idx]            # [B, H, W, 3]

        b_patch_min_x = batch_patch_min[:, 0]               # [B]
        b_patch_min_y = batch_patch_min[:, 1]               # [B]

        b_patch_max_x = b_patch_min_x + patch_width
        b_patch_max_y = b_patch_min_y + patch_height

        assert (b_patch_max_x <= self.width).all(), f"Some b_patch_max_x exceed self.width"
        assert (b_patch_max_y <= self.height).all(), f"Some b_patch_max_y exceed self.height"

        B = entire_ray_o.shape[0]

        # Create a grid of indices for the patches
        grid_y = th.arange(patch_height, device=device).view(1, patch_height, 1).expand(B, -1, patch_width)  # Shape: [B, PH, PW]
        grid_x = th.arange(patch_width, device=device).view(1, 1, patch_width).expand(B, patch_height, -1)   # Shape: [B, PH, PW]

        # Compute the absolute indices by adding the minimum patch indices
        y_idx = b_patch_min_y.view(B, 1, 1) + grid_y  # Shape: [B, PH, PW]
        x_idx = b_patch_min_x.view(B, 1, 1) + grid_x  # Shape: [B, PH, PW]

        # Create batch indices
        batch_idx = th.arange(B).view(B, 1, 1).expand(-1, patch_height, patch_width)  # Shape: [B, PH, PW]

        # Ensure indices are of type LongTensor
        batch_idx = batch_idx.long()
        y_idx = y_idx.long()
        x_idx = x_idx.long()

        # Use advanced indexing to extract the patches
        ray_o = entire_ray_o[batch_idx, y_idx, x_idx]  # Shape: [B, PH, PW, 3]
        ray_d = entire_ray_d[batch_idx, y_idx, x_idx]  # Shape: [B, PH, PW, 3]

        return ray_o, ray_d

    def forward(
        self,
        batch_mvp_idx: List[int],           # [B,]
        batch_patch_min: th.Tensor,         # [B, 2]
        patch_width: int,
        patch_height: int,

        verts: th.Tensor,                   # [P, 3]
        faces: th.Tensor,                   # [F, 3]
        verts_color: th.Tensor,             # [P, 3]
        faces_opacity: th.Tensor,           # [F,]
        faces_intense: th.Tensor,           # [B, F]

        background: th.Tensor,              # [3]

        aa_temperature: float = 1.0,
    ):
        
        '''
        Prepare inputs for the renderer
        '''
        num_batch = len(batch_mvp_idx)
        num_face = faces.shape[0]

        b_mv = self.mv[batch_mvp_idx]
        b_proj = self.proj[batch_mvp_idx]

        ### verts_ndc (B, P, 3), verts_image (B, P, 2)
        verts_ndc, verts_image = self.compute_verts_ndc_image(verts, b_mv, b_proj)

        ### aa structure
        aa_tris_verts = verts_image[:, faces.flatten()]    # [B, F*3, 2]
        aa_tris_verts = aa_tris_verts.view(-1, 3, 2)       # [B*F, 3, 2]
        aa_tris = Triangles(aa_tris_verts[:, 0], aa_tris_verts[:, 1], aa_tris_verts[:, 2])

        aa_face_verts = aa_tris.verts.reshape((num_batch, num_face, 3, 2))                  # [B, F, 3, 2]
        aa_face_edges = aa_tris.edges.reshape((num_batch, num_face, 3, 2))                  # [B, F, 3, 2]
        aa_face_edges_iszero = aa_tris.edges_iszero.reshape((num_batch, num_face, 3, 2))    # [B, F, 3, 2]
        aa_face_edges_recip = aa_tris.edges_recip.reshape((num_batch, num_face, 3, 2))      # [B, F, 3, 2]
        aa_face_edges_normal = aa_tris.edges_normal.reshape((num_batch, num_face, 3, 2))    # [B, F, 3, 2]
        aa_face_edges_normal_c = aa_tris.edges_normal_c.reshape((num_batch, num_face, 3))   # [B, F, 3]

        ### rays
        ray_o, ray_d = self.select_rays(batch_mvp_idx, batch_patch_min, patch_width, patch_height)

        color, depth = RenderFunction.apply(
            background.to(dtype=th.float32),

            batch_patch_min.to(dtype=th.int32),
            patch_width,
            patch_height,

            verts.to(dtype=th.float32),
            faces.to(dtype=th.int32),
            verts_color.to(dtype=th.float32),
            faces_opacity.to(dtype=th.float32),

            verts_ndc.to(dtype=th.float32),
            verts_image.to(dtype=th.float32),
            faces_intense.to(dtype=th.float32),

            aa_temperature,
            aa_face_verts.to(dtype=th.float32),
            aa_face_edges.to(dtype=th.float32),
            aa_face_edges_iszero.to(dtype=th.bool),
            aa_face_edges_recip.to(dtype=th.float32),
            aa_face_edges_normal.to(dtype=th.float32),
            aa_face_edges_normal_c.to(dtype=th.float32),
            self.aa_grad_buffer_size,

            ray_o.to(dtype=th.float32),
            ray_d.to(dtype=th.float32),
        )
        depth = (depth + 1.0) / 2.0     # normalize to [0, 1]
        depth = 1.0 - depth

        return color, depth

'''
===============================================================================
Render Layers
===============================================================================
'''

class LayeredRenderer(Renderer):

    def __init__(self, mv, proj, width, height, device):
        self.mv = mv                        # [B, 4, 4]
        self.proj = proj                    # [B, 4, 4]
        self.width = width
        self.height = height
        self.device = device

        self.num_batch = self.mv.shape[0]

        # initialize rays for each pixel
        self.ray_o = None
        self.ray_d = None
        self._init_rays()
        
    def generate(
        self,
        batch_mvp_idx: List[int],           # [B,]
        
        verts: th.Tensor,                   # [P, 3]
        faces: th.Tensor,                   # [F, 3]
        tets: th.Tensor,                    # [T, 4]
        face_tets: th.Tensor,               # [F, 2]
        tet_faces: th.Tensor,               # [T, 4]
        faces_existence: th.Tensor,         # [F,]

        num_layers: int,                    # int, L
    ):
        
        '''
        Prepare inputs for the renderer
        '''
        b_mv = self.mv[batch_mvp_idx]
        b_proj = self.proj[batch_mvp_idx]

        ### verts_ndc (B, P, 3), verts_image (B, P, 2)
        verts_ndc, verts_image = self.compute_verts_ndc_image(verts, b_mv, b_proj)

        ### rays
        ray_o, ray_d = self.ray_o[batch_mvp_idx], self.ray_d[batch_mvp_idx]

        width, height = self.width, self.height

        render_layers, render_layers_cnt = _C.generate_render_layers_cuda(
            width, height,

            verts.to(dtype=th.float32),
            faces.to(dtype=th.int32),
            tets.to(dtype=th.int32),
            face_tets.to(dtype=th.int32),
            tet_faces.to(dtype=th.int32),
            faces_existence.to(dtype=th.int32),

            verts_ndc.to(dtype=th.float32),
            verts_image.to(dtype=th.float32),

            ray_o.to(dtype=th.float32),
            ray_d.to(dtype=th.float32),

            num_layers
        )
        
        return render_layers, render_layers_cnt