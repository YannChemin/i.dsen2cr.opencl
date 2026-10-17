/* OpenCL C 1.1 kernels for i.dsen2cr.opencl.
 *
 * Activations are float32, channel-major (C, H, W) planes covering one
 * tile with its halo. Every convolution of DSen2-CR is 3 x 3, stride 1,
 * with zero padding 1, and is computed as an implicit matrix product
 *   out[cout][pixel] = sum_k wt[k][cout] * patch[k][pixel],
 * k = ci * 9 + ky * 3 + kx, the patch values being gathered from the
 * input planes on the fly. */

/* Work-group of WG x WG work-items, each computing RB x RB outputs: a
 * work-group computes a TILE_M (output channels) x TILE_N (pixels) block,
 * stepping through k by TILE_K. The host defines WG, RB and TILE_K at
 * build time (DSEN2CR_BUILD_OPTS in dsen2cr_model.h). WG * WG must be a
 * multiple of TILE_N and TILE_K a multiple of WG * WG / TILE_N. */
#if !defined(WG) || !defined(RB) || !defined(TILE_K)
#error "WG, RB and TILE_K must be defined at build time"
#endif
#define TILE_M (WG * RB)
#define TILE_N (WG * RB)
/* Patch rows loaded per work-item per step, and their spacing. */
#define KSTEP (WG * WG / TILE_N)
#define NLOAD (TILE_K / KSTEP)

/* Epilogue flags. */
#define EP_RELU 1
#define EP_RESIDUAL 2

/* 3 x 3 convolution with fused epilogue
 *   v = conv(in) + bias
 *   v = max(v, 0)            with EP_RELU
 *   v = res + scale * v      with EP_RESIDUAL
 * then v = 0 where the pixel lies outside the image, so that the next
 * layer sees the zero padding of a convolution over the whole image.
 *
 * in: cin planes of h x w. wt: (kpad, mpad) with kpad = cin * 9 rounded up
 * to TILE_K and mpad = cout rounded up to TILE_M, zero-filled. out and
 * res: cout planes of h x w. The tile pixel (x, y) is the image cell
 * (x0 + x, y0 + y) of an image of img_w x img_h cells.
 * Global size (ceil(h * w / TILE_N) * WG, mpad / TILE_M * WG), local size
 * (WG, WG). */
__kernel __attribute__((reqd_work_group_size(WG, WG, 1))) void
conv3x3(__global const float *in, int cin, __global const float *wt,
        __global const float *bias, __global float *out, int cout, int h,
        int w, int flags, float scale, __global const float *res, int x0,
        int y0, int img_w, int img_h)
{
    __local float ws[TILE_K][TILE_M];
    __local float xs[TILE_K][TILE_N];
    int tx = get_local_id(0), ty = get_local_id(1);
    int lid = ty * WG + tx;
    int n0 = get_group_id(0) * TILE_N, m0 = get_group_id(1) * TILE_M;
    int plane = h * w, ktot = cin * 9;
    int kpad = (ktot + TILE_K - 1) / TILE_K * TILE_K;
    int mpad = get_num_groups(1) * TILE_M;
    /* Each work-item loads one column of the patch block, which fixes the
     * pixel it gathers for. */
    int ln = lid % TILE_N, lk = lid / TILE_N;
    int pn = n0 + ln, py = pn / w, px = pn % w;
    /* Input channel and tap (ky * 3 + kx) of the patch rows this
     * work-item loads, advanced by TILE_K per step instead of divided
     * out: integer division is slow on GPUs. */
    int kc[NLOAD], kr[NLOAD];
    float acc[RB][RB];
    int i, j, k0, kk, l;

    for (i = 0; i < RB; i++)
        for (j = 0; j < RB; j++)
            acc[i][j] = 0.0f;
    for (l = 0; l < NLOAD; l++) {
        kc[l] = (lk + l * KSTEP) / 9;
        kr[l] = (lk + l * KSTEP) % 9;
    }

    for (k0 = 0; k0 < kpad; k0 += TILE_K) {
        for (l = 0; l < NLOAD; l++) {
            int kk = lk + l * KSTEP, k = k0 + kk;
            float v = 0.0f;

            ws[kk][ln] = wt[k * mpad + m0 + ln];
            if (k < ktot && pn < plane) {
                int dy = (kr[l] >= 3) + (kr[l] >= 6);
                int yy = py + dy - 1, xx = px + kr[l] - 3 * dy - 1;

                if (yy >= 0 && yy < h && xx >= 0 && xx < w)
                    v = in[kc[l] * plane + yy * w + xx];
            }
            xs[kk][ln] = v;
            kr[l] += TILE_K;
            while (kr[l] >= 9) {
                kr[l] -= 9;
                kc[l]++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (kk = 0; kk < TILE_K; kk++) {
            float a[RB], b[RB];

            for (i = 0; i < RB; i++)
                a[i] = ws[kk][ty + i * WG];
            for (j = 0; j < RB; j++)
                b[j] = xs[kk][tx + j * WG];
            for (i = 0; i < RB; i++)
                for (j = 0; j < RB; j++)
                    acc[i][j] = mad(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (j = 0; j < RB; j++) {
        int n = n0 + tx + j * WG;
        int yy, xx, inside;

        if (n >= plane)
            continue;
        yy = y0 + n / w;
        xx = x0 + n % w;
        inside = yy >= 0 && yy < img_h && xx >= 0 && xx < img_w;
        for (i = 0; i < RB; i++) {
            int m = m0 + ty + i * WG;
            float v;

            if (m >= cout)
                continue;
            v = acc[i][j] + bias[m];
            if (flags & EP_RELU)
                v = fmax(v, 0.0f);
            if (flags & EP_RESIDUAL)
                v = res[m * plane + n] + scale * v;
            out[m * plane + n] = inside ? v : 0.0f;
        }
    }
}
