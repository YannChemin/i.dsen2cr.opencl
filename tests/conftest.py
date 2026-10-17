"""pytest fixtures for i.dsen2cr.opencl.

The module always needs an OpenCL device (PoCL is enough). The tests use
small random DSen2-CR networks written with dsen2cr_convert.py, checked
against a NumPy implementation, so no published checkpoint is needed.
I_DSEN2CR_OPENCL_PLATFORM optionally selects the OpenCL platform.
"""

import os
import sys
from pathlib import Path

import grass.script as gs
import numpy as np
import pytest
from grass.tools import Tools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsen2cr_convert import write_safetensors  # noqa: E402

# A 10 m UTM grid of odd size, so that edge tiles are partial.
ROWS = 37
COLS = 29
RES = 10

BAND_KEYS = {
    "coastal": "B01",
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "nir09": "B09",
    "cirrus": "B10",
    "swir16": "B11",
    "swir22": "B12",
}
SUFFIXES = list(BAND_KEYS.values())

# Null cell of <s2_B04>, as (row, col).
NULL_CELL = (5, 7)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session on a UTM project with 13 synthetic L1C bands <s2_B*> in
    digital numbers, some above 10000 and below 0 to exercise clipping,
    SAR bands <s1_vv>/<s1_vh> in dB and a null cell in <s2_B04>."""
    project = tmp_path_factory.mktemp("i_dsen2cr_opencl") / "project"
    gs.create_project(project, epsg="32631")
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(
            n=5000000 + ROWS * RES,
            s=5000000,
            w=500000,
            e=500000 + COLS * RES,
            res=RES,
        )
        for i, band in enumerate(SUFFIXES):
            tools.r_mapcalc(
                expression=(
                    f"s2_{band} = 1500 + 400 * {i % 5} + "
                    f"9000 * sin(row() * {7 + i}) * cos(col() * {5 + i})"
                )
            )
        tools.r_mapcalc(
            expression=(
                f"s2_B04 = if(row() == {NULL_CELL[0] + 1} && "
                f"col() == {NULL_CELL[1] + 1}, null(), s2_B04)"
            ),
            overwrite=True,
        )
        tools.r_mapcalc(expression="s1_vv = -12 + 14 * sin(row() * 3 + col())")
        tools.r_mapcalc(expression="s1_vh = -20 + 15 * cos(row() - col() * 2)")
        tools.r_mapcalc(expression="s1_lin = 0.05 + 0.01 * row()")
        yield session


@pytest.fixture(scope="module")
def device_options():
    platform = os.environ.get("I_DSEN2CR_OPENCL_PLATFORM")
    return {"platform": platform} if platform else {}


def random_model(nblocks, feat, cin, seed=0, zero=False):
    """Random DSen2-CR weights, as dsen2cr_convert.py names them.

    :return: dict of tensor name to float32 array
    """
    rng = np.random.default_rng(seed)
    shapes = [("head", feat, cin)]
    for i in range(nblocks):
        shapes += [(f"blocks.{i}.conv1", feat, feat), (f"blocks.{i}.conv2", feat, feat)]
    shapes.append(("tail", 13, feat))
    tensors = {}
    for name, cout, c in shapes:
        bound = 0.0 if zero else np.sqrt(6.0 / (9 * c))
        tensors[f"{name}.weight"] = rng.uniform(-bound, bound, (cout, c, 3, 3)).astype(
            "<f4"
        )
        tensors[f"{name}.bias"] = (0.0 if zero else 0.1) * rng.standard_normal(
            cout
        ).astype("<f4")
    return tensors


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """Paths and weights of the test models: with SAR (B=2, F=8), optical
    only (B=1, F=136, over several output-channel blocks of the kernel
    with 64 or 128 channels per block) and an all-zero network, whose output is its clipped input."""
    root = tmp_path_factory.mktemp("models")
    out = {}
    for name, args in {
        "sar": (2, 8, 15),
        "opt": (1, 136, 13),
        "zero": (1, 4, 13),
    }.items():
        tensors = random_model(*args, seed=len(out), zero=name == "zero")
        path = root / f"{name}.safetensors"
        write_safetensors(path, tensors, {"format": "i.dsen2cr.opencl"})
        out[name] = (str(path), tensors)
    return out


def band_options(sar=True):
    """Input options for all bands, with the SAR bands when requested."""
    options = {key: f"s2_{band}" for key, band in BAND_KEYS.items()}
    if sar:
        options.update(vv="s1_vv", vh="s1_vh")
    return options


def conv3x3(x, weight, bias):
    """3 x 3 convolution with zero padding 1, (C, H, W) in and out."""
    _, h, w = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((weight.shape[0], h, w))
    for ky in range(3):
        for kx in range(3):
            out += np.einsum(
                "oc,chw->ohw", weight[:, :, ky, kx], xp[:, ky : ky + h, kx : kx + w]
            )
    return out + bias[:, None, None]


def reference(tensors, opt, sar=None):
    """NumPy DSen2-CR forward pass over the whole image.

    :param opt: (13, H, W) normalised optical input
    :param sar: (2, H, W) normalised SAR input or None
    :return: (13, H, W) normalised output
    """
    t = {k: v.astype(np.float64) for k, v in tensors.items()}
    x = opt if sar is None else np.concatenate([opt, sar])
    x = np.maximum(conv3x3(x, t["head.weight"], t["head.bias"]), 0)
    i = 0
    while f"blocks.{i}.conv1.weight" in t:
        y = np.maximum(
            conv3x3(x, t[f"blocks.{i}.conv1.weight"], t[f"blocks.{i}.conv1.bias"]), 0
        )
        x = x + 0.1 * conv3x3(
            y, t[f"blocks.{i}.conv2.weight"], t[f"blocks.{i}.conv2.bias"]
        )
        i += 1
    return opt + conv3x3(x, t["tail.weight"], t["tail.bias"])
