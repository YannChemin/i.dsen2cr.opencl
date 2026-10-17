"""Tests for i.dsen2cr.opencl."""

import numpy as np
import pytest
from conftest import (
    BAND_KEYS,
    COLS,
    NULL_CELL,
    ROWS,
    SUFFIXES,
    band_options,
    reference,
)
from grass.script import array as garray
from grass.tools import ToolError, Tools


def read(session, name):
    """Raster map as a float64 array, nulls as NaN."""
    return np.asarray(garray.array(name, null="nan", dtype=np.float64, env=session.env))


def normalised_inputs(session, sar):
    """Network inputs as the module builds them: optical DN clipped to
    [0, 10000] / 2000, SAR dB clipped and mapped to [0, 2], nulls filled
    with the band mean."""
    opt = []
    for band in SUFFIXES:
        a = read(session, f"s2_{band}")
        a[np.isnan(a)] = np.nanmean(a)
        opt.append(np.clip(a, 0, 10000) / 2000)
    opt = np.array(opt)
    if not sar:
        return opt, None
    vv = (np.clip(read(session, "s1_vv"), -25, 0) + 25) * 2 / 25
    vh = (np.clip(read(session, "s1_vh"), -32.5, 0) + 32.5) * 2 / 32.5
    return opt, np.array([vv, vh])


@pytest.fixture(scope="module")
def sar_run(session, models, device_options):
    """Run the SAR test model with 16-cell tiles: 3 x 3 tiles, the last
    ones partial, each with a 6-cell halo."""
    tools = Tools(session=session)
    region = tools.g_region(flags="g", format="json").json
    tools.i_dsen2cr_opencl(
        output="cr", model=models["sar"][0], tile=16, **band_options(), **device_options
    )
    return tools, region


def test_sar_model_matches_reference(session, models, sar_run):
    """Tiled inference equals the whole-image NumPy forward pass."""
    opt, sar = normalised_inputs(session, sar=True)
    expected = reference(models["sar"][1], opt, sar) * 2000
    expected[:, NULL_CELL[0], NULL_CELL[1]] = np.nan
    for i, band in enumerate(SUFFIXES):
        got = read(session, f"cr_{band}")
        np.testing.assert_allclose(got, expected[i], rtol=1e-4, atol=0.05, err_msg=band)


def test_tile_size_does_not_change_result(session, models, sar_run, device_options):
    """One tile over the whole region gives the same values as 16-cell
    tiles: the halo makes every tile core exact."""
    tools, _ = sar_run
    tools.i_dsen2cr_opencl(
        output="whole", model=models["sar"][0], **band_options(), **device_options
    )
    for band in ("B01", "B04", "B12"):
        np.testing.assert_allclose(
            read(session, f"whole_{band}"),
            read(session, f"cr_{band}"),
            rtol=1e-5,
            atol=0.01,
            err_msg=band,
        )


def test_null_input_cell_gives_null_output(session, sar_run):
    """A null cell in any input band is null in every output band, and
    no other cell is."""
    for band in SUFFIXES:
        got = read(session, f"cr_{band}")
        assert np.isnan(got).sum() == 1, band
        assert np.isnan(got[NULL_CELL]), band


def test_outputs_and_region(sar_run):
    """13 FCELL outputs on the region, which is not changed."""
    tools, before = sar_run
    maps = tools.g_list(type="raster", pattern="cr_*", format="json").json
    assert sorted(m["name"] for m in maps) == sorted(f"cr_{b}" for b in SUFFIXES)
    info = tools.r_info(map="cr_B02", format="json").json
    assert (info["rows"], info["cols"]) == (ROWS, COLS)
    assert info["datatype"] == "FCELL"
    assert tools.g_region(flags="g", format="json").json == before


def test_existing_output_needs_overwrite(session, models, sar_run, device_options):
    """Output maps are only replaced with --overwrite."""
    tools, _ = sar_run
    with pytest.raises(ToolError, match="already exists"):
        tools.i_dsen2cr_opencl(
            output="cr", model=models["sar"][0], **band_options(), **device_options
        )


def test_optical_model_matches_reference(session, models, device_options):
    """An optical-only model with more than 64 feature channels."""
    tools = Tools(session=session)
    tools.i_dsen2cr_opencl(
        output="opt",
        model=models["opt"][0],
        tile=20,
        **band_options(sar=False),
        **device_options,
    )
    opt, _ = normalised_inputs(session, sar=False)
    expected = reference(models["opt"][1], opt) * 2000
    for i, band in enumerate(SUFFIXES):
        got = read(session, f"opt_{band}")
        valid = ~np.isnan(got)
        np.testing.assert_allclose(
            got[valid], expected[i][valid], rtol=1e-4, atol=0.05, err_msg=band
        )


def test_scale_and_offset(session, models, device_options):
    """With a zero network the output is the clipped input, returned in
    the input units: reflectance = (value + offset) / scale."""
    tools = Tools(session=session)
    for band in SUFFIXES:
        tools.r_mapcalc(expression=f"refl_{band} = (s2_{band} - 1000) / 10000.")
    options = {key: f"refl_{band}" for key, band in BAND_KEYS.items()}
    tools.i_dsen2cr_opencl(
        output="zero",
        model=models["zero"][0],
        scale=1,
        offset=0.1,
        **options,
        **device_options,
    )
    for band in ("B02", "B11"):
        refl = read(session, f"refl_{band}")
        expected = np.clip(refl + 0.1, 0, 1) - 0.1
        expected[NULL_CELL] = np.nan  # Null in <refl_B04>, so in every band.
        np.testing.assert_allclose(read(session, f"zero_{band}"), expected, atol=1e-5)


def test_sar_model_needs_sar(session, models, device_options):
    """A SAR model refuses to run without the SAR bands."""
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="trained with SAR"):
        tools.i_dsen2cr_opencl(
            output="x",
            model=models["sar"][0],
            **band_options(sar=False),
            **device_options,
        )


def test_optical_model_refuses_sar(session, models, device_options):
    """SAR bands given to an optical-only model are refused, not ignored."""
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="without SAR"):
        tools.i_dsen2cr_opencl(
            output="x", model=models["opt"][0], **band_options(), **device_options
        )


def test_vv_and_vh_go_together(session, models, device_options):
    tools = Tools(session=session)
    options = band_options()
    del options["vh"]
    with pytest.raises(ToolError, match="together"):
        tools.i_dsen2cr_opencl(output="x", model=models["sar"][0], **options)


def test_linear_sar_is_refused(session, models, device_options):
    """Linear backscatter (positive mean) is refused: the network expects dB."""
    tools = Tools(session=session)
    options = band_options()
    options["vv"] = "s1_lin"
    with pytest.raises(ToolError, match="dB"):
        tools.i_dsen2cr_opencl(
            output="x", model=models["sar"][0], **options, **device_options
        )


def test_rejects_file_without_weights(session, tmp_path, device_options):
    """A safetensors file without DSen2-CR tensors fails loudly."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dsen2cr_convert import write_safetensors  # noqa: PLC0415

    path = tmp_path / "other.safetensors"
    write_safetensors(path, {"x": np.zeros(4, dtype="<f4")}, {})
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="no DSen2-CR weights"):
        tools.i_dsen2cr_opencl(
            output="x", model=str(path), **band_options(), **device_options
        )


def test_rejects_file_that_is_not_safetensors(session, tmp_path, device_options):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\xff" * 64)
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="not a valid safetensors"):
        tools.i_dsen2cr_opencl(
            output="x", model=str(path), **band_options(), **device_options
        )
