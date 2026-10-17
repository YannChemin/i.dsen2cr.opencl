## DESCRIPTION

*i.dsen2cr.opencl* removes clouds and cloud shadows from Sentinel-2
Level-1C images with DSen2-CR, the deep residual network of Meraner et
al. (2020). The network fuses the cloudy optical image with a Sentinel-1
SAR image of the same area, which sees through clouds, and predicts the
cloud-free reflectance of all 13 Sentinel-2 bands. It runs on an OpenCL
device (GPU or CPU) and reads the published model weights, converted
once to a safetensors file. Neither Python nor a deep learning framework
is needed at run time.

The 13 Sentinel-2 bands are given through their own options, named after
the STAC common band names: **coastal** (B01), **blue** (B02), **green**
(B03), **red** (B04), **rededge1** (B05), **rededge2** (B06),
**rededge3** (B07), **nir** (B08), **nir08** (B8A), **nir09** (B09),
**cirrus** (B10), **swir16** (B11) and **swir22** (B12). The Sentinel-1
backscatter is given with **vv** and **vh**.

The output maps are named after **output**, suffixed with the band name,
e.g. `output_B02`. They cover the current computational region, are in
the units of the optical input and get the colour table of the matching
input band. The network predicts every cell, cloudy or not: trained with
the cloud-adaptive regularized loss (CARL), it keeps clear cells close to
the input and reconstructs the cloudy ones.

### Input data

DSen2-CR is trained on the SEN12MS-CR dataset, and its inputs must match
that data:

- **Sentinel-2 Level-1C** top-of-atmosphere reflectance, all 13 bands
  including B10, resampled to the **10 m** grid of the computational
  region. Level-2A products lack B10 and are atmospherically corrected,
  unlike the training data. By default the input values are digital
  numbers, reflectance x 10000. **scale** and **offset** describe other
  units: reflectance = (value + **offset**) / **scale**. Products of
  processing baseline 04.00 and later need **offset**=-1000. The outputs
  are returned in the input units.
- **Sentinel-1** GRD backscatter (sigma0) **in dB**, VV and VH, on the
  same grid. The module refuses SAR maps whose mean is positive, which
  points to linear values. Convert them first, e.g.
  `r.mapcalc "vv_db = 10 * log10(vv)"`.

The network normalises its inputs as in training: optical digital
numbers are clipped to \[0, 10000\] and divided by 2000, VV is clipped to
\[-25, 0\] dB and VH to \[-32.5, 0\] dB, and both are mapped to \[0, 2\].

### Models

The authors published four checkpoints (Keras HDF5 files, see the
[DSen2-CR repository](https://github.com/ameraner/dsen2-cr)): the full
model trained on the CARL loss (recommended), the same network trained
on a plain L1 loss, and the two variants without SAR input. Convert a
checkpoint once with `dsen2cr_convert.py`, which needs the Python
package h5py (or PyTorch, for a state dict of the PyTorch port):

```sh
python3 dsen2cr_convert.py model_SARcarl.h5 dsen2cr_sar_carl.safetensors
```

The script is installed with the module in `etc/i.dsen2cr.opencl/` of
the GRASS installation (or of `$GRASS_ADDON_BASE` for an addon
installed with *g.extension*). The safetensors file is then given with
**model**.

The model tells which inputs are needed: **vv** and **vh** are
required by the SAR models and refused by the optical-only ones, so a
band is never silently ignored. The number of residual blocks and of
feature channels is read from the file (16 and 256 in the published
models).

## NOTES

### Tiles

The region is processed in square tiles of **tile** x **tile** cells.
Each tile is computed with a halo of 2 x blocks + 2 cells (34 cells for
the published models), which is the reach of the network. The tile
cores are therefore exactly what the network gives over the whole
region at once, with no seams. Larger tiles waste less computation in
the halos but need more device memory: about 1 GiB for 512-cell tiles
(the default) and 3.6 GiB for 1024-cell tiles with the published models.
If the tiles do not fit, the module stops and asks for a smaller
**tile**.

### Null cells

A cell that is null in any input band is null in every output band. For
the network input, null cells are filled with the mean of their band
over the region, as the reference implementation does with NaN values.
The fill only affects valid cells within 34 cells of the null ones.

### Performance

DSen2-CR is computationally heavy: about 39 million floating point
operations per cell for the published models. On an AMD Radeon Pro WX
7100 (Mesa Clover OpenCL) a 1024 x 1024 region takes about 30 seconds
with **tile**=1024, and a full Sentinel-2 granule (10980 x 10980 cells)
takes about 55 minutes. CPU devices are roughly 100 times slower (PoCL
on a 4-core Intel Core i7-1165G7: 45 seconds for a 64 x 64 region) and
only suited to small regions.

### Accuracy

The OpenCL implementation was checked against a PyTorch forward pass
(in double precision) of the published CARL model: the outputs agree to
within 0.015 digital numbers in every band.

## EXAMPLES

### Cloud removal with SAR

All bands imported at 10 m (e.g. with *i.sentinel.import* and
*r.resamp.interp*), Sentinel-1 backscatter in dB on the same grid:

```sh
g.region raster=T31TCJ_B02
i.dsen2cr.opencl coastal=T31TCJ_B01 blue=T31TCJ_B02 green=T31TCJ_B03 \
    red=T31TCJ_B04 rededge1=T31TCJ_B05 rededge2=T31TCJ_B06 \
    rededge3=T31TCJ_B07 nir=T31TCJ_B08 nir08=T31TCJ_B8A \
    nir09=T31TCJ_B09 cirrus=T31TCJ_B10 swir16=T31TCJ_B11 \
    swir22=T31TCJ_B12 vv=S1_VV_dB vh=S1_VH_dB offset=-1000 \
    model=$HOME/models/dsen2cr/dsen2cr_sar_carl.safetensors \
    output=T31TCJ_cr tile=1024
d.rgb red=T31TCJ_cr_B04 green=T31TCJ_cr_B03 blue=T31TCJ_cr_B02
```

### Keeping the clear cells

To replace only the cloudy cells, combine the result with a cloud mask
(here `clouds`: 1 for cloud and shadow, null elsewhere, e.g. from
*i.sentinel.mask*):

```sh
r.mapcalc "B04_filled = if(isnull(clouds), T31TCJ_B04, T31TCJ_cr_B04)"
```

## REFERENCES

Meraner, A., Ebel, P., Zhu, X. X., Schmitt, M., 2020. Cloud removal in
Sentinel-2 imagery using a deep residual neural network and SAR-optical
data fusion. ISPRS Journal of Photogrammetry and Remote Sensing 166,
333-346.
[doi:10.1016/j.isprsjprs.2020.05.013](https://doi.org/10.1016/j.isprsjprs.2020.05.013)

Ebel, P., Meraner, A., Schmitt, M., Zhu, X. X., 2020. Multisensor data
fusion for cloud removal in global and all-season Sentinel-2 imagery.
IEEE Transactions on Geoscience and Remote Sensing.
[doi:10.1109/TGRS.2020.3024744](https://doi.org/10.1109/TGRS.2020.3024744)
(the SEN12MS-CR dataset)

DSen2-CR source code and checkpoints:
[github.com/ameraner/dsen2-cr](https://github.com/ameraner/dsen2-cr)

## SEE ALSO

*[i.omnicloudmask](i.omnicloudmask.md),
[i.sen2sr.opencl](i.sen2sr.opencl.md),
[i.sentinel.import](i.sentinel.import.md),
[i.sentinel.mask](i.sentinel.mask.md),
[r.fillnulls](r.fillnulls.md)*

## AUTHORS

Yann Chemin
