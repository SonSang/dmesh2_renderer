#include <torch/extension.h>
#include <cstdio>
#include "render.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_forward_cuda", &RenderForwardCUDA);
    m.def("render_backward_cuda", &RenderBackwardCUDA);

    m.def("generate_render_layers_cuda", &GenerateRenderLayersCUDA);
}