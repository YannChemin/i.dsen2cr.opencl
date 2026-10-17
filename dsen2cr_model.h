#ifndef I_DSEN2CR_OPENCL_DSEN2CR_MODEL_H
#define I_DSEN2CR_OPENCL_DSEN2CR_MODEL_H

#include "ocl_backend.h"

/* Sentinel-2 L1C bands, in the order of the DSen2-CR inputs and outputs. */
#define S2_NBANDS 13

/* Sentinel-1 backscatter channels (VV, VH) of the SAR variants. */
#define S1_NBANDS 2

/* Blocking of the conv3x3 kernel: work-groups of WG x WG work-items,
 * each computing RB x RB outputs, stepping through the reduction by
 * TILE_K. The kernels are built with these values (DSEN2CR_BUILD_OPTS),
 * so host and device always agree. */
#define DSEN2CR_WG 16
#define DSEN2CR_RB 8
#define DSEN2CR_TILE_K 8
#define DSEN2CR_TILE (DSEN2CR_WG * DSEN2CR_RB)
#define DSEN2CR_STR2(x) #x
#define DSEN2CR_STR(x) DSEN2CR_STR2(x)
/* clang-format off */
#define DSEN2CR_BUILD_OPTS                                                     \
    "-DWG=" DSEN2CR_STR(DSEN2CR_WG)                                            \
    " -DRB=" DSEN2CR_STR(DSEN2CR_RB)                                           \
    " -DTILE_K=" DSEN2CR_STR(DSEN2CR_TILE_K)
/* clang-format on */

/* Residual scaling of every block, fixed by the DSen2-CR architecture. */
#define DSEN2CR_RES_SCALE 0.1f

struct dsen2cr_conv {
    int cin, cout;
    cl_mem w, b; /* Weights (kpad, mpad), see dsen2cr_kernels.cl; bias. */
};

/* DSen2-CR: a 3 x 3 head convolution with ReLU, nblocks residual blocks
 * (conv, ReLU, conv, scaled by 0.1, added to the block input), a 3 x 3
 * tail convolution back to the 13 bands and a long skip connection
 * adding the cloudy optical input. */
struct dsen2cr_model {
    int nsar;    /* 0 or S1_NBANDS. */
    int nin;     /* S2_NBANDS + nsar. */
    int feat;    /* Feature channels (F, 256 in the published models). */
    int nblocks; /* Residual blocks (B, 16 in the published models). */
    struct dsen2cr_conv head, tail, *conv; /* conv: 2 * nblocks. */

    struct ocl_backend *ocl;
    cl_kernel k_conv;

    /* Tile buffers for tile_w x tile_h cells (including the halo). */
    int tile_w, tile_h;
    cl_mem in, out, a, b, c;
};

/* Number of convolutions on the path from input to output, hence the
 * halo in cells that a tile needs for its core to be exact. */
#define DSEN2CR_HALO(m) (2 * (m)->nblocks + 2)

/* Load the weights written by dsen2cr_convert.py. Fatal on a missing
 * tensor or inconsistent shapes. */
void dsen2cr_model_load(struct dsen2cr_model *m, struct ocl_backend *ocl,
                        const char *path);

/* Device memory needed for tiles of w x h cells, in bytes. */
size_t dsen2cr_model_tile_bytes(const struct dsen2cr_model *m, int w, int h);

/* Allocate the device buffers for tiles of w x h cells. */
void dsen2cr_model_alloc(struct dsen2cr_model *m, int w, int h);

void dsen2cr_model_free(struct dsen2cr_model *m);

/* Run the network on one tile. in holds the nin normalised input planes
 * of tile_w x tile_h cells (zero outside the image), out receives the
 * S2_NBANDS output planes, normalised as the input. The tile's (0, 0)
 * cell is the image cell (x0, y0) of an img_w x img_h image. */
void dsen2cr_model_run(struct dsen2cr_model *m, const float *in, float *out,
                       int x0, int y0, int img_w, int img_h);

#endif /* I_DSEN2CR_OPENCL_DSEN2CR_MODEL_H */
