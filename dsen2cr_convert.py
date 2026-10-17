#!/usr/bin/env python3
"""Convert a DSen2-CR checkpoint to the safetensors file read by
i.dsen2cr.opencl.

MODULE:    i.dsen2cr.opencl
AUTHOR(S): Yann Chemin <dr.yann.chemin gmail.com>
PURPOSE:   Convert the published Keras (.h5) DSen2-CR checkpoints, or a
           state dict of the PyTorch port (.pth/.pt), to safetensors.
SPDX-License-Identifier: Unlicense

The output holds float32 tensors ``head.weight``/``head.bias``,
``blocks.<i>.conv1.*``, ``blocks.<i>.conv2.*`` and ``tail.*``, every
weight in the PyTorch layout (cout, cin, 3, 3). Only h5py (for .h5) or
PyTorch (for .pth/.pt) is needed.

Usage::

    python3 dsen2cr_convert.py model_SARcarl.h5 dsen2cr_sar_carl.safetensors
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def _decode(name):
    """Return an HDF5 attribute string as str."""
    return name.decode() if isinstance(name, bytes) else str(name)


def keras_convs(path):
    """Read the convolutions of a Keras DSen2-CR checkpoint, in layer order.

    Both ``model.save()`` files (weights under ``model_weights``) and
    ``model.save_weights()`` files are accepted. The weights of a nested
    model (e.g. saved through ``multi_gpu_model``) come in the order of
    its ``weight_names`` attribute.

    :param path: path of the .h5 file
    :return: list of (weight, bias), weight as (cout, cin, 3, 3)
    """
    import h5py  # noqa: PLC0415

    convs = []
    with h5py.File(path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        if "layer_names" not in root.attrs:
            sys.exit(f"{path}: not a Keras weight file (no layer_names)")
        kernel = None
        for layer in map(_decode, root.attrs["layer_names"]):
            group = root[layer]
            for wname in map(_decode, group.attrs.get("weight_names", [])):
                data = np.asarray(group[wname], dtype=np.float32)
                if wname.endswith("kernel:0"):
                    if data.ndim != 4 or data.shape[:2] != (3, 3):
                        sys.exit(f"{path}: {wname} is not a 3 x 3 kernel")
                    # Keras keeps (kh, kw, cin, cout) for channels_first too.
                    kernel = data.transpose(3, 2, 0, 1)
                elif wname.endswith("bias:0"):
                    if kernel is None:
                        sys.exit(f"{path}: bias {wname} without a kernel")
                    convs.append((kernel, data))
                    kernel = None
    return convs


def torch_convs(path):
    """Read the convolutions of a state dict of the PyTorch DSen2-CR port
    (``ResnetStackedArchitecture`` in dsen2cr_pytorch_model.py), in
    network order.

    :param path: path of the .pth/.pt file
    :return: list of (weight, bias), weight as (cout, cin, 3, 3)
    """
    import torch  # noqa: PLC0415

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    # Keys look like [module.][netResNet.]model.<i>[.conv_block.<j>].weight;
    # the integer path (i, j) orders the convolutions.
    weights = {}
    for key, value in state.items():
        match = re.search(r"model\.([\w.]+)\.(weight|bias)$", key)
        if match is None:
            continue
        index = tuple(int(i) for i in re.findall(r"\d+", match.group(1)))
        weights.setdefault(index, {})[match.group(2)] = value.float().numpy()
    convs = []
    for index in sorted(weights):
        entry = weights[index]
        if "weight" in entry and entry["weight"].ndim == 4:
            convs.append((entry["weight"], entry["bias"]))
    return convs


def check_and_name(convs, path):
    """Check that the convolutions form a DSen2-CR network and name them.

    :param convs: list of (weight, bias) in network order
    :param path: source path, for messages
    :return: dict of tensor name to array, and (nblocks, feat, cin)
    """
    if len(convs) < 2 or len(convs) % 2:
        sys.exit(f"{path}: {len(convs)} convolutions, expected 2 * B + 2")
    nblocks = (len(convs) - 2) // 2
    head_w = convs[0][0]
    feat, cin = head_w.shape[:2]
    if cin not in (13, 15):
        sys.exit(
            f"{path}: the first convolution reads {cin} channels, expected 13 or 15"
        )
    names = ["head"]
    for i in range(nblocks):
        names += [f"blocks.{i}.conv1", f"blocks.{i}.conv2"]
    names.append("tail")
    tensors = {}
    for name, (weight, bias) in zip(names, convs):
        want_in = cin if name == "head" else feat
        want_out = 13 if name == "tail" else feat
        if weight.shape != (want_out, want_in, 3, 3) or bias.shape != (want_out,):
            sys.exit(
                f"{path}: {name} has shape {weight.shape} / {bias.shape}, "
                f"expected ({want_out}, {want_in}, 3, 3) / ({want_out},)"
            )
        tensors[f"{name}.weight"] = np.ascontiguousarray(weight, dtype="<f4")
        tensors[f"{name}.bias"] = np.ascontiguousarray(bias, dtype="<f4")
    return tensors, (nblocks, feat, cin)


def write_safetensors(path, tensors, metadata):
    """Write float32 tensors to a safetensors file.

    :param path: output path
    :param tensors: dict of name to little-endian float32 array
    :param metadata: dict of str to str
    """
    header = {"__metadata__": metadata}
    offset = 0
    for name, array in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for array in tensors.values():
            f.write(array.tobytes())


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "checkpoint", type=Path, help="Keras .h5 or PyTorch .pth/.pt file"
    )
    parser.add_argument("output", type=Path, help="output .safetensors file")
    args = parser.parse_args()

    if args.checkpoint.suffix.lower() in (".h5", ".hdf5"):
        convs = keras_convs(args.checkpoint)
    else:
        convs = torch_convs(args.checkpoint)
    tensors, (nblocks, feat, cin) = check_and_name(convs, args.checkpoint)
    write_safetensors(
        args.output,
        tensors,
        {
            "format": "i.dsen2cr.opencl",
            "source": args.checkpoint.name,
            "blocks": str(nblocks),
            "features": str(feat),
            "sar": "yes" if cin == 15 else "no",
        },
    )
    print(
        f"{args.output}: {nblocks} blocks, {feat} features, "
        f"{'optical + SAR' if cin == 15 else 'optical only'} input"
    )


if __name__ == "__main__":
    main()
