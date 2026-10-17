#!/usr/bin/env python3
"""Download a co-registered Sentinel-1 / Sentinel-2 test set for
i.dsen2cr.opencl from the Copernicus Data Space Ecosystem (CDSE).

MODULE:    i.dsen2cr.opencl
AUTHOR(S): Yann Chemin <dr.yann.chemin gmail.com>
PURPOSE:   Find and download, on one 10 m UTM grid, a partly cloudy
           Sentinel-2 L1C scene (13 bands), the Sentinel-1 GRD backscatter
           (VV, VH in dB) closest in time and a clear Sentinel-2 L1C scene
           as reference, through the Sentinel Hub APIs of CDSE.
SPDX-License-Identifier: Unlicense

Steps:

1. List the Sentinel-2 L1C and Sentinel-1 GRD (IW, dual polarisation)
   acquisitions over the area with the Catalog API.
2. Screen every Sentinel-2 date at low resolution with the Sentinel Hub
   cloud mask (CLM): the cloud fraction and the data coverage inside the
   area, not over the whole granule.
3. Choose the cloudy date (cloud fraction in a range, preferring 40 %)
   with a Sentinel-1 date and a clear date close to it.
4. Download the three images at full resolution, in requests of at most
   2500 x 2500 cells, and write them as GeoTIFF files with a JSON summary.

Values: Sentinel-2 bands are L1C digital numbers (reflectance x 10000)
harmonised to processing baseline 03 (Sentinel Hub removes the -1000
offset of baseline 04.00 and later), so i.dsen2cr.opencl needs offset=0.
The 20 m and 60 m bands are resampled to 10 m by nearest neighbour.
Sentinel-1 is sigma0 (ellipsoid), orthorectified on the Copernicus 30 m
DEM, without speckle filter, in dB.

Authentication: CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (an OAuth client
from the Sentinel Hub dashboard of CDSE) when set, else the CDSE user
name and password of the ~/.netrc entry for
identity.dataspace.copernicus.eu.

Progress is reported to MetaMonitor
(https://github.com/YannChemin/MetaMonitor) when its spool exists.

Usage (defaults: Kanpur, 2025)::

    python3 cdse_s1s2_pair.py --out ~/RSDATA/Kanpur_DSen2CR
    python3 cdse_s1s2_pair.py --dry-run        # selection only
"""

import argparse
import datetime as dt
import io
import json
import math
import netrc
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_URL = f"{SH_URL}/api/v1/catalog/1.0.0/search"
PROCESS_URL = f"{SH_URL}/api/v1/process"

# Extent of the Kanpur Sentinel-1/-2 composites in ~/RSDATA/Kanpur_Sentinel_2025.
KANPUR_BBOX = (80.0763, 26.2635, 80.3275, 26.4882)

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
MAX_REQUEST = 2500  # Process API limit on output width and height.

EVAL_S2 = """//VERSION=3
function setup() {
  return {input: [{bands: %s, units: "DN"}],
          output: {bands: 13, sampleType: "UINT16"}};
}
function evaluatePixel(s) { return [%s]; }
""" % (json.dumps(S2_BANDS), ", ".join(f"s.{b}" for b in S2_BANDS))

EVAL_S2_MASK = """//VERSION=3
function setup() {
  return {input: [{bands: ["CLM", "CLP", "dataMask"]}],
          output: {bands: 3, sampleType: "UINT8"}};
}
function evaluatePixel(s) { return [s.CLM, s.CLP, s.dataMask]; }
"""

EVAL_S1 = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "dataMask"]}],
          output: {bands: 2, sampleType: "FLOAT32"}};
}
function db(v) { return v > 0 ? 10 * Math.log(v) / Math.LN10 : NaN; }
function evaluatePixel(s) {
  if (s.dataMask == 0) return [NaN, NaN];
  return [db(s.VV), db(s.VH)];
}
"""


class Progress:
    """Best-effort MetaMonitor slot (https://github.com/YannChemin/MetaMonitor).

    Writes the slot JSON atomically, at most once per second; never fails.
    """

    def __init__(self, slot, name):
        runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        self.dir = Path(os.environ.get("METAMONITOR_SPOOL", Path(runtime, "metamonitor", "slots")))
        self.slot, self.name, self.last = slot, name, 0.0

    def __call__(self, done, total, detail, state="running", force=False):
        now = time.monotonic()
        if not force and state == "running" and now - self.last < 1.0:
            return
        self.last = now
        data = {
            "name": self.name,
            "percent": 100.0 * done / total if total else None,
            "detail": detail,
            "state": state,
            "pid": os.getpid(),
        }
        try:
            if self.dir.is_dir():
                tmp = self.dir / f".{self.slot}.tmp"
                tmp.write_text(json.dumps(data))
                tmp.rename(self.dir / self.slot)
        except OSError:
            pass


class Session:
    """requests session with a CDSE OAuth token renewed before it expires."""

    def __init__(self):
        self.http = requests.Session()
        self.expires = 0.0
        client_id = os.environ.get("CDSE_CLIENT_ID")
        if client_id:
            self.form = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": os.environ.get("CDSE_CLIENT_SECRET", ""),
            }
        else:
            auth = netrc.netrc().authenticators("identity.dataspace.copernicus.eu")
            if auth is None:
                sys.exit(
                    "No CDSE credentials: set CDSE_CLIENT_ID/CDSE_CLIENT_SECRET or add "
                    "identity.dataspace.copernicus.eu to ~/.netrc"
                )
            self.form = {
                "grant_type": "password",
                "client_id": "cdse-public",
                "username": auth[0],
                "password": auth[2],
            }

    def _token(self):
        if time.time() < self.expires - 60:
            return
        r = requests.post(TOKEN_URL, data=self.form, timeout=60)
        if r.status_code != 200:
            sys.exit(f"CDSE authentication failed ({r.status_code}): {r.text[:300]}")
        tok = r.json()
        self.http.headers["Authorization"] = f"Bearer {tok['access_token']}"
        self.expires = time.time() + tok.get("expires_in", 600)

    def post(self, url, payload, accept="application/json"):
        """POST JSON with retries on rate limiting and server errors."""
        for attempt in range(6):
            self._token()
            r = self.http.post(url, json=payload, headers={"Accept": accept}, timeout=600)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            if r.status_code != 200:
                sys.exit(f"{url}: HTTP {r.status_code}: {r.text[:500]}")
            return r
        sys.exit(f"{url}: giving up after repeated HTTP {r.status_code}")


def utm_grid(bbox, res):
    """UTM CRS and 10 m-aligned extent covering a lon/lat bbox.

    :return: (epsg, (xmin, ymin, xmax, ymax), cols, rows)
    """
    from pyproj import Transformer  # noqa: PLC0415

    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    xs, ys = tr.transform(
        [bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]]
    )
    xmin = math.floor(min(xs) / res) * res
    ymin = math.floor(min(ys) / res) * res
    xmax = math.ceil(max(xs) / res) * res
    ymax = math.ceil(max(ys) / res) * res
    return epsg, (xmin, ymin, xmax, ymax), round((xmax - xmin) / res), round((ymax - ymin) / res)


def catalog(session, collection, bbox, start, end):
    """All Catalog API features of a collection over bbox and dates."""
    payload = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 100,
    }
    features = []
    while True:
        page = session.post(CATALOG_URL, payload, accept="application/geo+json").json()
        features += page.get("features", [])
        nxt = page.get("context", {}).get("next")
        if nxt is None:
            return features
        payload["next"] = nxt


def by_date(features):
    """Group catalog features by UTC acquisition date."""
    dates = {}
    for f in features:
        day = f["properties"]["datetime"][:10]
        dates.setdefault(day, []).append(f)
    return dates


def s1_usable(feature):
    """IW acquisitions with both VV and VH."""
    p = feature["properties"]
    mode = p.get("sar:instrument_mode", "IW")
    pols = p.get("sar:polarizations") or p.get("s1:polarization") or ["VV", "VH"]
    if isinstance(pols, str):
        pols = ["VV", "VH"] if pols.upper() in ("DV", "VV+VH", "VV,VH") else [pols]
    return mode == "IW" and "VV" in pols and "VH" in pols


def process(session, epsg, extent, width, height, data, evalscript):
    """One Process API request returning a GeoTIFF, as (bands, rows, cols)."""
    import rasterio  # noqa: PLC0415

    payload = {
        "input": {
            "bounds": {
                "bbox": list(extent),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
            },
            "data": data,
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    r = session.post(PROCESS_URL, payload, accept="image/tiff")
    with rasterio.MemoryFile(io.BytesIO(r.content)) as mem, mem.open() as ds:
        return ds.read()


def s2_data(day, **processing):
    return [
        {
            "type": "sentinel-2-l1c",
            "dataFilter": {
                "timeRange": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"},
                "mosaickingOrder": "leastCC",
            },
            "processing": {"harmonizeValues": True, **processing},
        }
    ]


def s1_data(day):
    return [
        {
            "type": "sentinel-1-grd",
            "dataFilter": {
                "timeRange": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"},
                "acquisitionMode": "IW",
                "polarization": "DV",
                "resolution": "HIGH",
                "mosaickingOrder": "mostRecent",
            },
            "processing": {
                "backCoeff": "SIGMA0_ELLIPSOID",
                "orthorectify": True,
                "demInstance": "COPERNICUS_30",
            },
        }
    ]


def screen(session, epsg, extent, dates, screen_res, progress):
    """Cloud fraction and coverage of each Sentinel-2 date inside the area."""
    width = max(1, math.ceil((extent[2] - extent[0]) / screen_res))
    height = max(1, math.ceil((extent[3] - extent[1]) / screen_res))
    stats = {}
    for i, day in enumerate(sorted(dates)):
        progress(i, len(dates), f"screening Sentinel-2 {day}")
        clm, _, valid = process(session, epsg, extent, width, height, s2_data(day), EVAL_S2_MASK)
        nvalid = int((valid > 0).sum())
        stats[day] = {
            "coverage": nvalid / valid.size,
            "cloud_fraction": float((clm[valid > 0] > 0).mean()) if nvalid else 1.0,
            "granule_cloud_cover": [f["properties"].get("eo:cloud_cover") for f in dates[day]],
        }
        print(
            f"  {day}: coverage {stats[day]['coverage']:6.1%}, "
            f"cloud {stats[day]['cloud_fraction']:6.1%}",
            flush=True,
        )
    return stats


def choose(stats, s1_dates, args):
    """Best (cloudy, clear, s1) date triple, or None.

    Prefers a cloud fraction near 40 %, then short gaps to the SAR and the
    clear date.
    """

    def days(a, b):
        return abs((dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days)

    full = {d: s for d, s in stats.items() if s["coverage"] >= args.min_coverage}
    clear = [d for d, s in full.items() if s["cloud_fraction"] <= args.clear_max]
    best = None
    for day, s in full.items():
        if not args.cloudy_min <= s["cloud_fraction"] <= args.cloudy_max:
            continue
        s1 = min(s1_dates, key=lambda d: days(d, day), default=None)
        ref = min((c for c in clear if c != day), key=lambda d: days(d, day), default=None)
        if s1 is None or ref is None:
            continue
        if days(s1, day) > args.max_s1_gap or days(ref, day) > args.max_clear_gap:
            continue
        score = abs(s["cloud_fraction"] - 0.4) + 0.03 * days(s1, day) + 0.01 * days(ref, day)
        if best is None or score < best[0]:
            best = (score, day, ref, s1)
    return None if best is None else best[1:]


def download(session, epsg, extent, cols, rows, res, data, evalscript, nbands, dtype, progress, what):
    """Full-resolution image assembled from requests of <= MAX_REQUEST cells."""
    out = np.zeros((nbands, rows, cols), dtype=dtype)
    tiles = [
        (r0, c0)
        for r0 in range(0, rows, MAX_REQUEST)
        for c0 in range(0, cols, MAX_REQUEST)
    ]
    for k, (r0, c0) in enumerate(tiles):
        progress(k, len(tiles), f"{what}: block {k + 1}/{len(tiles)}", force=True)
        h = min(MAX_REQUEST, rows - r0)
        w = min(MAX_REQUEST, cols - c0)
        ext = (
            extent[0] + c0 * res,
            extent[3] - (r0 + h) * res,
            extent[0] + (c0 + w) * res,
            extent[3] - r0 * res,
        )
        out[:, r0 : r0 + h, c0 : c0 + w] = process(session, epsg, ext, w, h, data, evalscript)
    return out


def write_tif(path, array, epsg, extent, res, descriptions, nodata=None):
    import rasterio  # noqa: PLC0415
    from rasterio.transform import from_origin  # noqa: PLC0415

    profile = {
        "driver": "GTiff",
        "width": array.shape[2],
        "height": array.shape[1],
        "count": array.shape[0],
        "dtype": array.dtype,
        "crs": f"EPSG:{epsg}",
        "transform": from_origin(extent[0], extent[3], res, res),
        "compress": "deflate",
        "tiled": True,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array)
        for i, d in enumerate(descriptions, 1):
            ds.set_band_description(i, d)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bbox", type=float, nargs=4, default=KANPUR_BBOX, metavar=("W", "S", "E", "N"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out", type=Path, default=Path.home() / "RSDATA" / "Kanpur_DSen2CR")
    parser.add_argument("--res", type=float, default=10.0, help="output cell size (m)")
    parser.add_argument("--screen-res", type=float, default=160.0, help="screening cell size (m, at most 200)")
    parser.add_argument("--cloudy-min", type=float, default=0.2)
    parser.add_argument("--cloudy-max", type=float, default=0.7)
    parser.add_argument("--clear-max", type=float, default=0.02)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--max-s1-gap", type=int, default=3, help="days")
    parser.add_argument("--max-clear-gap", type=int, default=20, help="days")
    parser.add_argument("--cloudy-date", help="force the cloudy Sentinel-2 date (YYYY-MM-DD)")
    parser.add_argument("--clear-date", help="force the clear Sentinel-2 date")
    parser.add_argument("--s1-date", help="force the Sentinel-1 date")
    parser.add_argument("--dry-run", action="store_true", help="stop after the selection")
    args = parser.parse_args()

    progress = Progress("cdse-s1s2-pair", "CDSE S1/S2 pair download")
    session = Session()
    epsg, extent, cols, rows = utm_grid(args.bbox, args.res)
    print(f"Grid: EPSG:{epsg}, {cols} x {rows} cells of {args.res:g} m, extent {extent}")

    forced = args.cloudy_date and args.clear_date and args.s1_date
    if forced:
        cloudy, clear, s1 = args.cloudy_date, args.clear_date, args.s1_date
        stats = {}
    else:
        progress(0, 1, "catalog search", force=True)
        s2 = by_date(catalog(session, "sentinel-2-l1c", args.bbox, args.start, args.end))
        s1_all = by_date(catalog(session, "sentinel-1-grd", args.bbox, args.start, args.end))
        s1_dates = sorted(d for d, fs in s1_all.items() if any(s1_usable(f) for f in fs))
        print(f"Catalog: {len(s2)} Sentinel-2 L1C dates, {len(s1_dates)} Sentinel-1 IW DV dates")
        # Granules fully clouded or clear everywhere need no screening.
        candidates = {
            d: fs
            for d, fs in s2.items()
            if min(f["properties"].get("eo:cloud_cover", 0) for f in fs) < 98
        }
        print(f"Screening {len(candidates)} dates at {args.screen_res:g} m...")
        stats = screen(session, epsg, extent, candidates, args.screen_res, progress)
        picked = choose(stats, s1_dates, args)
        if picked is None:
            progress(1, 1, "no suitable date triple", state="failed", force=True)
            sys.exit(
                "No cloudy date with a Sentinel-1 date and a clear date close enough; "
                "widen --cloudy-min/--cloudy-max, --max-s1-gap or --max-clear-gap"
            )
        cloudy, clear, s1 = picked
        cloudy = args.cloudy_date or cloudy
        clear = args.clear_date or clear
        s1 = args.s1_date or s1
    print(
        f"Selected: cloudy S2 {cloudy} "
        f"({stats.get(cloudy, {}).get('cloud_fraction', float('nan')):.1%} cloud), "
        f"clear S2 {clear}, S1 {s1}"
    )
    if args.dry_run:
        progress(1, 1, f"selected {cloudy} / {clear} / {s1}", state="done", force=True)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("s2_l1c_cloudy", cloudy, s2_data(cloudy), EVAL_S2, 13, np.uint16, S2_BANDS, 0),
        ("s2_l1c_clear", clear, s2_data(clear), EVAL_S2, 13, np.uint16, S2_BANDS, 0),
        ("s2_cloudmask_cloudy", cloudy, s2_data(cloudy), EVAL_S2_MASK, 3, np.uint8, ["CLM", "CLP", "dataMask"], None),
        ("s1_grd_db", s1, s1_data(s1), EVAL_S1, 2, np.float32, ["VV", "VH"], float("nan")),
    ]
    files = {}
    for i, (stem, day, data, script, nb, dtype, desc, nodata) in enumerate(jobs):
        what = f"{stem} {day}"
        print(f"Downloading {what}...")
        sub = Progress("cdse-s1s2-pair", "CDSE S1/S2 pair download")

        def step(done, total, detail, state="running", force=False, i=i):
            sub(i + done / total, len(jobs), detail, state, force)

        array = download(session, epsg, extent, cols, rows, args.res, data, script, nb, dtype, step, what)
        path = args.out / f"{stem}_{day.replace('-', '')}.tif"
        write_tif(path, array, epsg, extent, args.res, desc, nodata)
        files[stem] = path.name

    summary = {
        "bbox_lonlat": list(args.bbox),
        "epsg": epsg,
        "extent": list(extent),
        "cols": cols,
        "rows": rows,
        "res": args.res,
        "cloudy_date": cloudy,
        "clear_date": clear,
        "s1_date": s1,
        "screening": stats,
        "files": files,
        "s2_units": "L1C DN = reflectance x 10000, harmonised (offset 0)",
        "s1_units": "sigma0 ellipsoid, orthorectified (COPERNICUS_30), dB, no speckle filter",
    }
    (args.out / "pair.json").write_text(json.dumps(summary, indent=2))
    progress(1, 1, f"done: {cloudy} / {clear} / {s1}", state="done", force=True)
    print(f"Done; summary in {args.out / 'pair.json'}")


if __name__ == "__main__":
    main()
