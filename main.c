/****************************************************************************
 *
 * MODULE:       i.dsen2cr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Remove clouds from Sentinel-2 L1C imagery with the DSen2-CR
 *               residual network and SAR-optical fusion (Meraner et al.
 *               2020) on an OpenCL device.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "dsen2cr_model.h"
#include "ocl_backend.h"

/* Input normalisation of the DSen2-CR training (dsen2cr_main.py):
 * optical digital numbers (reflectance x 10000) clipped to [0, 10000]
 * and divided by 2000; SAR backscatter in dB clipped to [-25, 0] (VV)
 * and [-32.5, 0] (VH), then mapped linearly to [0, 2]. */
#define OPT_DN_MAX 10000.0
#define OPT_SCALE 2000.0
#define SAR_MAX_VAL 2.0
static const double sar_clip_min[S1_NBANDS] = {-25.0, -32.5};
static const double sar_clip_max[S1_NBANDS] = {0.0, 0.0};

static const char *band_suffix[S2_NBANDS] = {"B01", "B02", "B03", "B04", "B05",
                                             "B06", "B07", "B08", "B8A", "B09",
                                             "B10", "B11", "B12"};

static const char *band_key[S2_NBANDS] = {
    "coastal", "blue",  "green", "red",    "rededge1", "rededge2", "rededge3",
    "nir",     "nir08", "nir09", "cirrus", "swir16",   "swir22"};

static const char *band_desc[S2_NBANDS] = {
    "Sentinel-2 B01 (coastal aerosol, 60 m)",
    "Sentinel-2 B02 (blue, 10 m)",
    "Sentinel-2 B03 (green, 10 m)",
    "Sentinel-2 B04 (red, 10 m)",
    "Sentinel-2 B05 (red edge 1, 20 m)",
    "Sentinel-2 B06 (red edge 2, 20 m)",
    "Sentinel-2 B07 (red edge 3, 20 m)",
    "Sentinel-2 B08 (NIR, 10 m)",
    "Sentinel-2 B8A (narrow NIR, 20 m)",
    "Sentinel-2 B09 (water vapour, 60 m)",
    "Sentinel-2 B10 (SWIR cirrus, 60 m)",
    "Sentinel-2 B11 (SWIR 1.6 um, 20 m)",
    "Sentinel-2 B12 (SWIR 2.2 um, 20 m)"};

/* Warn when the region is not on a 10 m grid: the SEN12MS-CR training
 * patches hold all bands resampled to 10 m. */
static void check_resolution(const struct Cell_head *win)
{
    double f = G_database_units_to_meters_factor();

    if (G_projection() == PROJECTION_LL || f <= 0.0)
        G_warning(_("Unable to check that the region resolution is 10 m in "
                    "this coordinate system; DSen2-CR is trained on "
                    "Sentinel-2 bands resampled to 10 m"));
    else if (fabs(win->ns_res * f - 10.0) > 0.01 ||
             fabs(win->ew_res * f - 10.0) > 0.01)
        G_warning(_("The region resolution is %g x %g m, but DSen2-CR is "
                    "trained on Sentinel-2 bands resampled to 10 m"),
                  win->ew_res * f, win->ns_res * f);
}

/* Mean of the non-null cells of each map over the region, used to fill
 * null cells in the network input as the reference implementation does
 * (their output is null). Fatal on a map without any value. */
static void band_means(const int *fd, const char **names, int n, int rows,
                       int cols, double *mean)
{
    FCELL *buf = Rast_allocate_f_buf();
    double sum[S2_NBANDS + S1_NBANDS] = {0};
    long cnt[S2_NBANDS + S1_NBANDS] = {0};
    int i, r, c;

    G_message(_("Computing band means..."));
    for (r = 0; r < rows; r++) {
        G_percent(r, rows, 5);
        for (i = 0; i < n; i++) {
            Rast_get_f_row(fd[i], buf, r);
            for (c = 0; c < cols; c++)
                if (!Rast_is_f_null_value(&buf[c])) {
                    sum[i] += buf[c];
                    cnt[i]++;
                }
        }
    }
    G_percent(1, 1, 1);
    for (i = 0; i < n; i++) {
        if (cnt[i] == 0)
            G_fatal_error(_("Raster map <%s> has only null cells in the "
                            "current region"),
                          names[i]);
        mean[i] = sum[i] / cnt[i];
    }
    G_free(buf);
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct Option *band_opt[S2_NBANDS], *vv_opt, *vh_opt, *output, *model_opt,
        *scale_opt, *offset_opt, *tile_opt, *device, *platform;
    struct ocl_backend ocl;
    struct dsen2cr_model model;
    struct Cell_head win;
    const char *in_names[S2_NBANDS + S1_NBANDS];
    char out_names[S2_NBANDS][GNAME_MAX];
    int in_fd[S2_NBANDS + S1_NBANDS], out_fd[S2_NBANDS];
    double mean[S2_NBANDS + S1_NBANDS];
    FCELL *row_buf, *out_row;
    float *strip, *tile_in, *tile_out, *out_strip;
    char *null_strip;
    int H, W, T, th, tw, halo, sh, tile_w, tile_h, nin, nty, ntx, ty, tx;
    double scale, offset, dn_factor;
    size_t tile_plane, need;
    time_t t_start;
    int i, r, c;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("cloud removal"));
    G_add_keyword(_("Sentinel"));
    G_add_keyword(_("SAR"));
    G_add_keyword(_("fusion"));
    G_add_keyword(_("deep learning"));
    G_add_keyword(_("GPU"));
    G_add_keyword(_("OpenCL"));
    module->description =
        _("Removes clouds from Sentinel-2 L1C imagery with the DSen2-CR "
          "network and Sentinel-1 SAR-optical fusion on an OpenCL device.");

    for (i = 0; i < S2_NBANDS; i++) {
        band_opt[i] = G_define_standard_option(G_OPT_R_INPUT);
        band_opt[i]->key = band_key[i];
        band_opt[i]->description = _(band_desc[i]);
        band_opt[i]->guisection = _("Optical");
    }

    vv_opt = G_define_standard_option(G_OPT_R_INPUT);
    vv_opt->key = "vv";
    vv_opt->required = NO;
    vv_opt->label = _("Sentinel-1 VV backscatter in dB");
    vv_opt->description = _("Required by the models trained with SAR input");
    vv_opt->guisection = _("SAR");

    vh_opt = G_define_standard_option(G_OPT_R_INPUT);
    vh_opt->key = "vh";
    vh_opt->required = NO;
    vh_opt->label = _("Sentinel-1 VH backscatter in dB");
    vh_opt->description = _("Required by the models trained with SAR input");
    vh_opt->guisection = _("SAR");

    output = G_define_standard_option(G_OPT_R_BASENAME_OUTPUT);
    output->description =
        _("Basename for output raster maps, suffixed with the band name "
          "(e.g. _B02)");

    model_opt = G_define_standard_option(G_OPT_F_INPUT);
    model_opt->key = "model";
    model_opt->label = _("DSen2-CR weights (safetensors)");
    model_opt->description =
        _("Converted from a published DSen2-CR checkpoint with "
          "dsen2cr_convert.py");
    model_opt->guisection = _("Model");

    scale_opt = G_define_option();
    scale_opt->key = "scale";
    scale_opt->type = TYPE_DOUBLE;
    scale_opt->answer = "10000";
    scale_opt->label = _("Scale factor of the optical input values");
    scale_opt->description =
        _("Reflectance = (value + offset) / scale; use 1 for inputs "
          "already in reflectance");
    scale_opt->guisection = _("Optical");

    offset_opt = G_define_option();
    offset_opt->key = "offset";
    offset_opt->type = TYPE_DOUBLE;
    offset_opt->answer = "0";
    offset_opt->label =
        _("Offset added to the optical input values before scaling");
    offset_opt->description =
        _("E.g. -1000 for products of processing baseline 04.00 or later; "
          "outputs are returned in the input units");
    offset_opt->guisection = _("Optical");

    tile_opt = G_define_option();
    tile_opt->key = "tile";
    tile_opt->type = TYPE_INTEGER;
    tile_opt->answer = "512";
    tile_opt->options = "16-8192";
    tile_opt->label = _("Side of the output tile in cells");
    tile_opt->description =
        _("Each tile is computed with a halo of 2 * blocks + 2 cells on every "
          "side; larger tiles waste less work on halos but need more device "
          "memory");
    tile_opt->guisection = _("OpenCL");

    device = G_define_option();
    device->key = "device";
    device->type = TYPE_STRING;
    device->options = "auto,gpu,cpu";
    device->answer = "auto";
    device->description = _("OpenCL device type");
    device->guisection = _("OpenCL");

    platform = G_define_option();
    platform->key = "platform";
    platform->type = TYPE_STRING;
    platform->required = NO;
    platform->description =
        _("Only use OpenCL platforms whose name contains this string "
          "(case-insensitive)");
    platform->guisection = _("OpenCL");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    scale = atof(scale_opt->answer);
    offset = atof(offset_opt->answer);
    T = atoi(tile_opt->answer);
    if (scale <= 0.0)
        G_fatal_error(_("Option <%s> must be positive"), scale_opt->key);
    if ((vv_opt->answer != NULL) != (vh_opt->answer != NULL))
        G_fatal_error(_("Options <%s> and <%s> must be given together"),
                      vv_opt->key, vh_opt->key);

    for (i = 0; i < S2_NBANDS; i++) {
        snprintf(out_names[i], GNAME_MAX, "%s_%s", output->answer,
                 band_suffix[i]);
        if (G_legal_filename(out_names[i]) < 0)
            G_fatal_error(_("<%s> is an illegal file name"), out_names[i]);
        if (G_find_raster2(out_names[i], G_mapset()) &&
            !G_check_overwrite(argc, argv))
            G_fatal_error(_("Raster map <%s> already exists"), out_names[i]);
    }

    ocl_init(&ocl, device->answer, platform->answer, DSEN2CR_BUILD_OPTS);
    dsen2cr_model_load(&model, &ocl, model_opt->answer);
    G_message(_("DSen2-CR model: %d residual blocks, %d features, %s"),
              model.nblocks, model.feat,
              model.nsar ? _("optical and SAR input") : _("optical input"));

    /* The SAR inputs are used exactly when the model reads them. */
    if (model.nsar && !vv_opt->answer)
        G_fatal_error(_("The model <%s> was trained with SAR input: options "
                        "<%s> and <%s> are required"),
                      model_opt->answer, vv_opt->key, vh_opt->key);
    if (!model.nsar && vv_opt->answer)
        G_fatal_error(_("The model <%s> was trained without SAR input: "
                        "remove options <%s> and <%s>, or use a SAR model"),
                      model_opt->answer, vv_opt->key, vh_opt->key);
    nin = model.nin;
    for (i = 0; i < S2_NBANDS; i++)
        in_names[i] = band_opt[i]->answer;
    if (model.nsar) {
        in_names[S2_NBANDS] = vv_opt->answer;
        in_names[S2_NBANDS + 1] = vh_opt->answer;
    }

    G_get_window(&win);
    check_resolution(&win);
    H = win.rows;
    W = win.cols;

    for (i = 0; i < nin; i++)
        in_fd[i] = Rast_open_old(in_names[i], "");

    band_means(in_fd, in_names, nin, H, W, mean);
    /* Backscatter in dB is almost always negative; a positive mean means
     * linear power or amplitude, which the network was not trained on. */
    for (i = S2_NBANDS; i < nin; i++)
        if (mean[i] > 0.0)
            G_fatal_error(_("The mean of <%s> is %g: SAR inputs must be "
                            "backscatter in dB (10 * log10 of sigma0), not "
                            "linear values"),
                          in_names[i], mean[i]);

    /* Tiles: cores of th x tw cells partitioning the region, each run
     * with a halo so that its core equals whole-image inference. */
    halo = DSEN2CR_HALO(&model);
    th = T < H ? T : H;
    tw = T < W ? T : W;
    tile_h = th + 2 * halo;
    tile_w = tw + 2 * halo;
    nty = (H + th - 1) / th;
    ntx = (W + tw - 1) / tw;
    need = dsen2cr_model_tile_bytes(&model, tile_w, tile_h);
    if (need > ocl.global_mem ||
        sizeof(float) * (size_t)tile_w * tile_h * model.feat > ocl.max_alloc)
        G_fatal_error(_("Tiles of %d x %d cells (with the halo) need %lu MiB "
                        "of device memory and buffers of %lu MiB; <%s> has "
                        "%lu MiB with buffers up to %lu MiB: reduce option "
                        "<%s>"),
                      tile_w, tile_h, (unsigned long)(need >> 20),
                      (unsigned long)((sizeof(float) * (size_t)tile_w * tile_h *
                                       model.feat) >>
                                      20),
                      ocl.device_name, (unsigned long)(ocl.global_mem >> 20),
                      (unsigned long)(ocl.max_alloc >> 20), tile_opt->key);
    dsen2cr_model_alloc(&model, tile_w, tile_h);
    G_message(_("%d x %d tiles of %d x %d cells, halo %d cells, %lu MiB of "
                "device memory"),
              ntx, nty, tw, th, halo, (unsigned long)(need >> 20));

    for (i = 0; i < S2_NBANDS; i++)
        out_fd[i] = Rast_open_fp_new(out_names[i]);

    /* Optical values to digital numbers (reflectance x 10000). */
    dn_factor = OPT_DN_MAX / scale;
    sh = th + 2 * halo;
    tile_plane = (size_t)tile_w * tile_h;
    row_buf = Rast_allocate_f_buf();
    out_row = Rast_allocate_f_buf();
    strip = G_malloc(sizeof(float) * nin * sh * W);
    null_strip = G_malloc((size_t)th * W);
    tile_in = G_malloc(sizeof(float) * nin * tile_plane);
    tile_out = G_malloc(sizeof(float) * S2_NBANDS * tile_plane);
    out_strip = G_malloc(sizeof(float) * S2_NBANDS * th * W);

    G_message(_("Removing clouds..."));
    t_start = time(NULL);
    for (ty = 0; ty < nty; ty++) {
        int y0 = ty * th, rows = y0 + th <= H ? th : H - y0;

        /* Read the core rows and the halo rows around them, normalised,
         * with nulls replaced by the band mean; zero outside the image. */
        memset(null_strip, 0, (size_t)th * W);
        for (r = 0; r < sh; r++) {
            int src = y0 - halo + r;

            for (i = 0; i < nin; i++) {
                float *dst = strip + ((size_t)i * sh + r) * W;

                if (src < 0 || src >= H) {
                    memset(dst, 0, sizeof(float) * W);
                    continue;
                }
                Rast_get_f_row(in_fd[i], row_buf, src);
                for (c = 0; c < W; c++) {
                    double v;
                    int is_null = Rast_is_f_null_value(&row_buf[c]);

                    if (is_null && r >= halo && r < halo + th)
                        null_strip[(size_t)(r - halo) * W + c] = 1;
                    if (i < S2_NBANDS) {
                        v = is_null ? mean[i] : row_buf[c];
                        v = (v + offset) * dn_factor;
                        v = v < 0.0 ? 0.0 : v > OPT_DN_MAX ? OPT_DN_MAX : v;
                        dst[c] = (float)(v / OPT_SCALE);
                    }
                    else {
                        int s = i - S2_NBANDS;
                        double lo = sar_clip_min[s], hi = sar_clip_max[s];

                        v = is_null ? mean[i] : row_buf[c];
                        v = v < lo ? lo : v > hi ? hi : v;
                        dst[c] = (float)(SAR_MAX_VAL * (v - lo) / (hi - lo));
                    }
                }
            }
        }

        for (tx = 0; tx < ntx; tx++) {
            int x0 = tx * tw, cols = x0 + tw <= W ? tw : W - x0;

            G_percent(ty * ntx + tx, nty * ntx, 1);
            for (i = 0; i < nin; i++)
                for (r = 0; r < tile_h; r++) {
                    const float *src = strip + ((size_t)i * sh + r) * W;
                    float *dst = tile_in + i * tile_plane + (size_t)r * tile_w;

                    for (c = 0; c < tile_w; c++) {
                        int sc = x0 - halo + c;

                        dst[c] = sc >= 0 && sc < W ? src[sc] : 0.0f;
                    }
                }

            dsen2cr_model_run(&model, tile_in, tile_out, x0 - halo, y0 - halo,
                              W, H);

            for (i = 0; i < S2_NBANDS; i++)
                for (r = 0; r < rows; r++)
                    memcpy(out_strip + ((size_t)i * th + r) * W + x0,
                           tile_out + i * tile_plane +
                               (size_t)(r + halo) * tile_w + halo,
                           sizeof(float) * cols);
        }

        for (i = 0; i < S2_NBANDS; i++)
            for (r = 0; r < rows; r++) {
                const float *src = out_strip + ((size_t)i * th + r) * W;
                const char *nulls = null_strip + (size_t)r * W;

                for (c = 0; c < W; c++) {
                    if (nulls[c])
                        Rast_set_f_null_value(&out_row[c], 1);
                    else
                        out_row[c] =
                            (FCELL)(src[c] * OPT_SCALE / dn_factor - offset);
                }
                Rast_put_f_row(out_fd[i], out_row);
            }
    }
    G_percent(1, 1, 1);
    G_verbose_message(_("Inference took %ld s"), (long)(time(NULL) - t_start));

    for (i = 0; i < nin; i++)
        Rast_close(in_fd[i]);
    for (i = 0; i < S2_NBANDS; i++) {
        struct History hist;
        struct Colors colors;
        char title[256];

        Rast_close(out_fd[i]);
        snprintf(title, sizeof(title),
                 "Sentinel-2 %s, clouds removed by "
                 "DSen2-CR",
                 band_suffix[i]);
        Rast_put_cell_title(out_names[i], title);
        if (Rast_read_colors(band_opt[i]->answer, "", &colors) > 0) {
            Rast_write_colors(out_names[i], G_mapset(), &colors);
            Rast_free_colors(&colors);
        }
        Rast_short_history(out_names[i], "raster", &hist);
        Rast_append_format_history(&hist, "Model: %s", model_opt->answer);
        Rast_command_history(&hist);
        Rast_write_history(out_names[i], &hist);
    }

    G_free(row_buf);
    G_free(out_row);
    G_free(strip);
    G_free(null_strip);
    G_free(tile_in);
    G_free(tile_out);
    G_free(out_strip);
    dsen2cr_model_free(&model);
    ocl_free(&ocl);

    G_message(_("Written %d raster maps <%s_B01> to <%s_B12>"), S2_NBANDS,
              output->answer, output->answer);
    exit(EXIT_SUCCESS);
}
