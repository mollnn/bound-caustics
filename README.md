# Bernstein Bounds for Caustics

Code for SIGGRAPH 2025 (ToG) paper "Bernstein Bounds for Caustics"

[[Paper]](https://zhiminfan.work/paper/bound_caustics_preprint.pdf)
[[Supplementary]](https://zhiminfan.work/paper/bound_caustics_preprint_supp.pdf)
[[Video (Compressed)]](https://zhiminfan.work/video/bound_caustics_compressed.mp4)

## Build

The implementation includes the precomputation end and the rendering end. 

- Precomputation: We have Python implementation for all cases, which is easy to run. Only single scattering case has C++ implementation. Double scattering is speeded up using Numba. 

- Rendering: We based on the code of specular polynomials. I copied the building instructions below:

    > The project is based on https://github.com/VicentChen/mitsuba. Please install the dependency first. (I provide a precompiled version for msvc at https://github.com/mollnn/mitsuba0.6-dep-py3.9.12)
    >```
    >cd mts1
    >mkdir cbuild
    >cd cbuild
    >cmake ..
    >```
    >Then build the generated project in cbuild. Tested on Windows 10, Visual Studio 2022. The implementation builds upon Mbglints and CyPolynomials.

## Reproduce

I plan to release code for almost all figures this time. Unfortunately, some experiments rely on local code modifications, so put them all together needs time.

- **Fig. 09: demonstration of how to convert position/irradiance bounds into distributions.** In the directory `2d/fig09`, run `run_main_top/bottom.py`. You can choose the `_latex` variant to generate the paper figure, which however needs a latex environment. 

    - Bonus: To visualize the computation process of polynomials expressed in Bernstein basis, please try `2d/mid/run_main.py`. We show the polynomials (curves) and their control points (dots).
    ![result](2d/mid/result.png) *This figure is not shown in the paper because of some formatting inconvenience. Nevertheless, I still feel it's helpful for understanding.*

- **Fig. 10: ablation on multi-sample estimators.** Please run `test/fig_plane/cmp_sample.py`. 

- **Fig. 13 (Top): main experiment (single reflection).** Please run `test/fig_plane/test.py`. 

## Acknowledgement

Some scenes are modified from [SMS](https://github.com/tizian/specular-manifold-sampling). The implementation builds upon [Mbglints](https://github.com/wangningbei/mbglints) and [CyPolynomials](http://codebase.cemyuksel.com/code.html). We sincerely thank the authors for kindly release their code and scenes, as well as their great works.