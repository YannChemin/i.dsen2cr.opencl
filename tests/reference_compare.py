"""Compare i.dsen2cr.opencl with a PyTorch forward pass of the same weights.

Developer check, not part of the pytest suite: it needs PyTorch and a
converted DSen2-CR checkpoint (any size, e.g. the published 16 x 256
network). Synthetic L1C and SAR bands with a cloud disc are written on the
current region, the module is run on them with the given extra options,
and every output band is compared with the PyTorch result over the whole
region (zero padding at the region edges, as the module).

Usage (inside a GRASS session on a metric project; keep the region small,
the PyTorch pass runs on the CPU):

    g.region n=5000960 s=5000000 w=500000 e=500960 res=10
    python3 reference_compare.py MODEL.safetensors [module options ...]
"""

import sys

import grass.script as gs
import numpy as np
from grass.script import array as garray

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
KEYS = [
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",
    "cirrus",
    "swir16",
    "swir22",
]
BASE = [1400, 1200, 1100, 1000, 1300, 2000, 2300, 2400, 2500, 800, 20, 1900, 1200]


def read(name):
    """Raster map as float64 array."""
    return np.asarray(garray.array(name, null="nan", dtype=np.float64))


def torch_forward(path, x):
    """DSen2-CR forward pass of the safetensors weights on (C, H, W) input."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    w = {k: v.double() for k, v in load_file(path).items()}
    inp = torch.from_numpy(x)[None]
    with torch.no_grad():
        h = F.relu(F.conv2d(inp, w["head.weight"], w["head.bias"], padding=1))
        i = 0
        while f"blocks.{i}.conv1.weight" in w:
            t = F.relu(F.conv2d(h, w[f"blocks.{i}.conv1.weight"], w[f"blocks.{i}.conv1.bias"], padding=1))
            h = h + 0.1 * F.conv2d(t, w[f"blocks.{i}.conv2.weight"], w[f"blocks.{i}.conv2.bias"], padding=1)
            i += 1
        out = inp[:, :13] + F.conv2d(h, w["tail.weight"], w["tail.bias"], padding=1)
    return out[0].numpy()


def main():
    model = sys.argv[1]
    extra = dict(arg.split("=", 1) for arg in sys.argv[2:])
    region = gs.region()
    cy, cx = region["rows"] / 2, region["cols"] / 2
    disc = f"sqrt((row() - {cy})^2 + (col() - {cx})^2) < {min(cy, cx) / 3}"
    for band, base in zip(BANDS, BASE):
        gs.mapcalc(
            f"refcmp_{band} = {base} * (1 + 0.3 * sin(row() / 9.) * cos(col() / 13.))"
            f" + if({disc}, 6000, 0)",
            overwrite=True,
        )
    gs.mapcalc("refcmp_vv = -10 + 3 * sin(row() / 7.) * cos(col() / 5.)", overwrite=True)
    gs.mapcalc("refcmp_vh = -17 + 3 * cos(row() / 6.) * sin(col() / 11.)", overwrite=True)

    options = {key: f"refcmp_{band}" for key, band in zip(KEYS, BANDS)}
    with open(model, "rb") as f:
        header = f.read(4096)
    use_sar = b'"head.weight":{"dtype":"F32","shape":[' in header and b",15,3,3]" in header
    if use_sar:
        options.update(vv="refcmp_vv", vh="refcmp_vh")
    gs.run_command(
        "i.dsen2cr.opencl", output="refcmp_out", model=model, overwrite=True, **options, **extra
    )

    opt = np.array([np.clip(read(f"refcmp_{b}"), 0, 10000) / 2000 for b in BANDS])
    x = opt
    if use_sar:
        vv = (np.clip(read("refcmp_vv"), -25, 0) + 25) * 2 / 25
        vh = (np.clip(read("refcmp_vh"), -32.5, 0) + 32.5) * 2 / 32.5
        x = np.concatenate([opt, [vv, vh]])
    expected = torch_forward(model, x) * 2000

    worst = 0.0
    for i, band in enumerate(BANDS):
        got = read(f"refcmp_out_{band}")
        diff = np.abs(got - expected[i])
        worst = max(worst, diff.max())
        print(f"{band}: max |diff| {diff.max():.4f} DN, mean {diff.mean():.5f} DN")
    print(f"worst max |diff| over all bands: {worst:.4f} DN")


if __name__ == "__main__":
    main()
