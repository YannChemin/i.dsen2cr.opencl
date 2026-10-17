# i.dsen2cr.opencl

GRASS GIS addon that removes clouds from Sentinel-2 Level-1C images with
[DSen2-CR](https://github.com/ameraner/dsen2-cr) (Meraner et al. 2020), a
deep residual network fusing the optical image with Sentinel-1 SAR, on an
OpenCL device (GPU or CPU). It is a C + OpenCL reimplementation of the
network's inference; neither Python nor a deep learning framework is
needed at run time.

See [i.dsen2cr.opencl.md](i.dsen2cr.opencl.md) for the full manual.

## Requirements

- GRASS GIS 8.x source tree (for building the addon)
- An OpenCL 1.1 implementation with single-precision support and
  work-groups of 256 work-items, e.g. Mesa Clover/rusticl, a vendor GPU
  driver, or PoCL for CPUs
- OpenCL headers and ICD loader (`libOpenCL`), e.g. Debian packages
  `opencl-headers` and `ocl-icd-opencl-dev`
- A DSen2-CR checkpoint (see below) and, once for its conversion, the
  Python package h5py

## Build and install

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

or `g.extension extension=i.dsen2cr.opencl url=/path/to/i.dsen2cr.opencl`.
The OpenCL kernels in `dsen2cr_kernels.cl` are embedded into the binary
at build time.

## Getting the model

The authors' checkpoints are linked from the
[DSen2-CR README](https://github.com/ameraner/dsen2-cr#trained-model-checkpoints)
(Google Drive): CARL loss with SAR (recommended), L1 loss with SAR, and
both without SAR. Convert one to safetensors:

```sh
python3 dsen2cr_convert.py model_SARcarl.h5 dsen2cr_sar_carl.safetensors
```

## Usage

```sh
g.region raster=T31TCJ_B02    # all bands on the 10 m grid
i.dsen2cr.opencl coastal=B01 blue=B02 green=B03 red=B04 rededge1=B05 \
    rededge2=B06 rededge3=B07 nir=B08 nir08=B8A nir09=B09 cirrus=B10 \
    swir16=B11 swir22=B12 vv=S1_VV_dB vh=S1_VH_dB offset=-1000 \
    model=dsen2cr_sar_carl.safetensors output=cr tile=1024
```

The optical inputs are L1C digital numbers (reflectance x 10000; use
`offset=-1000` for processing baseline 04.00 and later), the SAR inputs
sigma0 in dB. Outputs are FCELL maps `<output>_B01` ... `<output>_B12`.

## Tests

Tests use pytest with small random networks checked against a NumPy
implementation, so no checkpoint is needed; they need an OpenCL device
(PoCL is enough) and the installed module:

```sh
export I_DSEN2CR_OPENCL_PLATFORM=Clover  # optional
PYTHONPATH=$(grass --config python_path) python3 -m pytest tests/
```

`tests/reference_compare.py` compares the module with a PyTorch forward
pass of a real checkpoint (developer check, needs PyTorch and
safetensors); run it inside a GRASS session on a small metric region:

```sh
python3 tests/reference_compare.py dsen2cr_sar_carl.safetensors [options]
```

With the published CARL model the outputs agree to 0.015 digital numbers.

## Reference

Meraner, A., Ebel, P., Zhu, X. X., Schmitt, M., 2020. Cloud removal in
Sentinel-2 imagery using a deep residual neural network and SAR-optical
data fusion. ISPRS Journal of Photogrammetry and Remote Sensing 166,
333-346. <https://doi.org/10.1016/j.isprsjprs.2020.05.013>

## License

Code: [Unlicense](LICENSE). The DSen2-CR checkpoints are distributed
separately by their authors, under their terms.

## Author

Yann Chemin
