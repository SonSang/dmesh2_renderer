This repository contains rendering code for [DMesh++](https://github.com/SonSang/dmesh2). The code is based on the implementation of [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and [dmesh_renderer](https://github.com/SonSang/dmesh_renderer). There are two different renderers here.

* `Renderer`: It renders a set of (semi-transparent) triangles efficiently with global depth testing. It generates visibility gradients by using Anti-Aliasing.
* `LayeredRenderer`: It renders a list of layers with strict depth ordering. That is, the first layer contains the rendering of the first triangles that each pixel ray collides. Then, the second layer contains the information of the next triangles that each pixel ray collides after the first ones, and so on. This is used for a situation where the point positions of `DMesh++` are fixed.