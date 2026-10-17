#!/bin/bash
# MODULE:    i.dsen2cr.opencl
# AUTHOR(S): Yann Chemin <dr.yann.chemin gmail.com>
# PURPOSE:   Import a test set written by cdse_s1s2_pair.py into a new GRASS
#            project, run i.dsen2cr.opencl on it and compare the result with
#            the clear reference scene, inside and outside the clouds.
# SPDX-License-Identifier: Unlicense
#
# Usage: evaluate_pair.sh PAIR_DIR PROJECT_PATH MODEL [i.dsen2cr.opencl options]
#   e.g. evaluate_pair.sh ~/RSDATA/Kanpur_DSen2CR ~/grassdata/kanpur_dsen2cr \
#            ~/models/dsen2cr/dsen2cr_sar_carl.safetensors tile=1024 platform=Clover
#
# The module reports its progress to MetaMonitor through G_percent()
# (https://github.com/YannChemin/MetaMonitor).

set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '9,12p' "$0"
    exit 1
fi
PAIR=$(realpath "$1")
PROJECT=$2
MODEL=$(realpath "$3")
shift 3
EXTRA="$*"

json() { python3 -I -c "import json,sys; print(json.load(open('$PAIR/pair.json'))$1)"; }
CLOUDY=$PAIR/$(json "['files']['s2_l1c_cloudy']")
CLEAR=$PAIR/$(json "['files']['s2_l1c_clear']")
MASK=$PAIR/$(json "['files']['s2_cloudmask_cloudy']")
S1=$PAIR/$(json "['files']['s1_grd_db']")

if [ ! -d "$PROJECT" ]; then
    grass -c "$CLOUDY" "$PROJECT" -e
fi

grass "$PROJECT/PERMANENT" --exec bash -s "$CLOUDY" "$CLEAR" "$MASK" "$S1" "$MODEL" "$EXTRA" <<'EOF'
set -euo pipefail
CLOUDY=$1 CLEAR=$2 MASK=$3 S1=$4 MODEL=$5 EXTRA=$6
export LC_NUMERIC=C  # printf %f expects a decimal point.
BANDS=(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12)
KEYS=(coastal blue green red rededge1 rededge2 rededge3 nir nir08 nir09 cirrus swir16 swir22)

# Fail loudly on a CRS-less or empty project.
g.proj -p | grep -q -i "utm" || { echo "Project has no UTM CRS" >&2; exit 1; }
r.in.gdal --q --o input="$CLOUDY" output=cloudy
r.in.gdal --q --o input="$CLEAR" output=clear
r.in.gdal --q --o input="$MASK" band=1 output=clm
r.in.gdal --q --o input="$MASK" band=3 output=datamask
r.in.gdal --q --o input="$S1" output=s1
g.region raster=cloudy.1 -p

OPTS=""
for i in "${!BANDS[@]}"; do
    OPTS="$OPTS ${KEYS[$i]}=cloudy.$((i + 1))"
    r.null --q map=cloudy.$((i + 1)) setnull=0
done
i.dsen2cr.opencl --o --v $OPTS vv=s1.1 vh=s1.2 model="$MODEL" output=cr $EXTRA

# Cloudy cells from the Sentinel Hub cloud mask (CLM) of the cloudy scene.
r.mapcalc --q --o "cloud = if(datamask == 1 && clm == 1, 1, null())"
r.mapcalc --q --o "clearsky = if(datamask == 1 && clm == 0, 1, null())"
# Opaque cloud: mean visible TOA reflectance of the cloudy scene above 0.30.
r.mapcalc --q --o "opaque = if(datamask == 1 && (cloudy.2 + cloudy.3 + cloudy.4) / 3. > 3000, 1, null())"
echo
count() { r.univar -g "$1" | sed -n 's/^n=//p'; }
echo "Cells: opaque $(count opaque), cloud mask $(count cloud), clear $(count clearsky)"
echo "Mean absolute difference to the clear scene (DN), input / output:"
printf "%-4s %22s %22s %22s\n" band opaque "cloud mask" clear
mean() { r.univar -g "$1" | sed -n 's/^mean=//p'; }
for i in "${!BANDS[@]}"; do
    b=$((i + 1))
    for zone in opaque cloud clearsky; do
        r.mapcalc --q --o "e_in_$zone = if(isnull($zone), null(), abs(cloudy.$b - clear.$b))"
        r.mapcalc --q --o "e_out_$zone = if(isnull($zone), null(), abs(cr_${BANDS[$i]} - clear.$b))"
    done
    printf "%-4s %10.1f / %9.1f %10.1f / %9.1f %10.1f / %9.1f\n" "${BANDS[$i]}" \
        "$(mean e_in_opaque)" "$(mean e_out_opaque)" \
        "$(mean e_in_cloud)" "$(mean e_out_cloud)" \
        "$(mean e_in_clearsky)" "$(mean e_out_clearsky)"
done
EOF
