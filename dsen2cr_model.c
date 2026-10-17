/****************************************************************************
 *
 * MODULE:       i.dsen2cr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      DSen2-CR network (Meraner et al. 2020) on an OpenCL
 *               device: weight loading and tile inference.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "dsen2cr_model.h"
#include "safetensors.h"

#define WG DSEN2CR_WG
#define TILE_M DSEN2CR_TILE
#define TILE_N DSEN2CR_TILE
#define TILE_K DSEN2CR_TILE_K

#define EP_RELU 1
#define EP_RESIDUAL 2

/* Load one convolution, checking its shape, and upload its weights
 * transposed to (k, cout) with k = ci * 9 + ky * 3 + kx, zero-padded to
 * the kernel blocking. cin or cout of -1 accept any extent. */
static struct dsen2cr_conv load_conv(struct ocl_backend *ocl,
                                     const struct st_file *f,
                                     const char *prefix, int cin, int cout)
{
    struct dsen2cr_conv c;
    char name[128];
    long shape[4] = {cout, cin, 3, 3}, bshape[1];
    float *w, *b, *wt, *bp;
    int ktot, kpad, mpad, m, k;

    snprintf(name, sizeof(name), "%s.weight", prefix);
    w = st_get_f32(f, name, 4, shape);
    c.cout = (int)shape[0];
    c.cin = (int)shape[1];
    snprintf(name, sizeof(name), "%s.bias", prefix);
    bshape[0] = c.cout;
    b = st_get_f32(f, name, 1, bshape);

    ktot = c.cin * 9;
    kpad = (ktot + TILE_K - 1) / TILE_K * TILE_K;
    mpad = (c.cout + TILE_M - 1) / TILE_M * TILE_M;
    wt = G_calloc((size_t)kpad * mpad, sizeof(float));
    for (m = 0; m < c.cout; m++)
        for (k = 0; k < ktot; k++)
            wt[(size_t)k * mpad + m] = w[(size_t)m * ktot + k];
    bp = G_calloc(mpad, sizeof(float));
    memcpy(bp, b, sizeof(float) * c.cout);

    c.w = ocl_upload(ocl, wt, sizeof(float) * kpad * mpad, prefix);
    c.b = ocl_upload(ocl, bp, sizeof(float) * mpad, prefix);
    G_free(w);
    G_free(b);
    G_free(wt);
    G_free(bp);
    return c;
}

void dsen2cr_model_load(struct dsen2cr_model *m, struct ocl_backend *ocl,
                        const char *path)
{
    struct st_file f;
    char prefix[64];
    size_t wgs;
    int i;

    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    st_open(&f, path);

    if (!st_find(&f, "head.weight"))
        G_fatal_error(_("<%s> holds no DSen2-CR weights (tensor <%s> not "
                        "found); convert the checkpoint with "
                        "dsen2cr_convert.py"),
                      path, "head.weight");

    m->head = load_conv(ocl, &f, "head", -1, -1);
    m->feat = m->head.cout;
    m->nin = m->head.cin;
    m->nsar = m->nin - S2_NBANDS;
    if (m->nsar != 0 && m->nsar != S1_NBANDS)
        G_fatal_error(_("<%s>: the first convolution reads %d channels, "
                        "expected %d (optical only) or %d (optical and SAR)"),
                      path, m->nin, S2_NBANDS, S2_NBANDS + S1_NBANDS);

    for (m->nblocks = 0;; m->nblocks++) {
        snprintf(prefix, sizeof(prefix), "blocks.%d.conv1.weight", m->nblocks);
        if (!st_find(&f, prefix))
            break;
    }
    m->conv = G_malloc(sizeof(struct dsen2cr_conv) * 2 * m->nblocks);
    for (i = 0; i < m->nblocks; i++) {
        snprintf(prefix, sizeof(prefix), "blocks.%d.conv1", i);
        m->conv[2 * i] = load_conv(ocl, &f, prefix, m->feat, m->feat);
        snprintf(prefix, sizeof(prefix), "blocks.%d.conv2", i);
        m->conv[2 * i + 1] = load_conv(ocl, &f, prefix, m->feat, m->feat);
    }
    m->tail = load_conv(ocl, &f, "tail", m->feat, S2_NBANDS);
    st_close(&f);

    m->k_conv = ocl_kernel(ocl, "conv3x3");
    ocl_check(clGetKernelWorkGroupInfo(m->k_conv, ocl->device,
                                       CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs),
                                       &wgs, NULL),
              "clGetKernelWorkGroupInfo");
    if (wgs < WG * WG)
        G_fatal_error(_("OpenCL: device <%s> runs at most %lu work-items per "
                        "work-group for the convolution kernel, %d are "
                        "needed"),
                      ocl->device_name, (unsigned long)wgs, WG * WG);
}

size_t dsen2cr_model_tile_bytes(const struct dsen2cr_model *m, int w, int h)
{
    size_t plane = sizeof(float) * w * h;

    return plane * (m->nin + S2_NBANDS + 3 * (size_t)m->feat);
}

void dsen2cr_model_alloc(struct dsen2cr_model *m, int w, int h)
{
    struct ocl_backend *ocl = m->ocl;
    size_t plane = sizeof(float) * w * h;

    m->tile_w = w;
    m->tile_h = h;
    m->in = ocl_alloc(ocl, plane * m->nin, "input tile");
    m->out = ocl_alloc(ocl, plane * S2_NBANDS, "output tile");
    m->a = ocl_alloc(ocl, plane * m->feat, "features A");
    m->b = ocl_alloc(ocl, plane * m->feat, "features B");
    m->c = ocl_alloc(ocl, plane * m->feat, "features C");
}

static void free_conv(struct dsen2cr_conv *c)
{
    ocl_release(&c->w);
    ocl_release(&c->b);
}

void dsen2cr_model_free(struct dsen2cr_model *m)
{
    int i;

    free_conv(&m->head);
    free_conv(&m->tail);
    for (i = 0; i < 2 * m->nblocks; i++)
        free_conv(&m->conv[i]);
    G_free(m->conv);
    ocl_release(&m->in);
    ocl_release(&m->out);
    ocl_release(&m->a);
    ocl_release(&m->b);
    ocl_release(&m->c);
    if (m->k_conv)
        clReleaseKernel(m->k_conv);
    memset(m, 0, sizeof(*m));
}

struct tile_pos {
    int x0, y0, img_w, img_h;
};

static void conv(struct dsen2cr_model *m, const struct dsen2cr_conv *c,
                 cl_mem in, cl_mem out, int flags, float scale, cl_mem res,
                 const struct tile_pos *p)
{
    cl_kernel k = m->k_conv;
    int h = m->tile_h, w = m->tile_w, a = 0;
    int mpad = (c->cout + TILE_M - 1) / TILE_M * TILE_M;
    size_t local[2] = {WG, WG};
    size_t global[2];

    global[0] = (size_t)((h * w + TILE_N - 1) / TILE_N) * WG;
    global[1] = (size_t)(mpad / TILE_M) * WG;
    if (!res)
        res = out; /* Unused, but must be a valid buffer. */

    clSetKernelArg(k, a++, sizeof(cl_mem), &in);
    clSetKernelArg(k, a++, sizeof(int), &c->cin);
    clSetKernelArg(k, a++, sizeof(cl_mem), &c->w);
    clSetKernelArg(k, a++, sizeof(cl_mem), &c->b);
    clSetKernelArg(k, a++, sizeof(cl_mem), &out);
    clSetKernelArg(k, a++, sizeof(int), &c->cout);
    clSetKernelArg(k, a++, sizeof(int), &h);
    clSetKernelArg(k, a++, sizeof(int), &w);
    clSetKernelArg(k, a++, sizeof(int), &flags);
    clSetKernelArg(k, a++, sizeof(float), &scale);
    clSetKernelArg(k, a++, sizeof(cl_mem), &res);
    clSetKernelArg(k, a++, sizeof(int), &p->x0);
    clSetKernelArg(k, a++, sizeof(int), &p->y0);
    clSetKernelArg(k, a++, sizeof(int), &p->img_w);
    ocl_check(clSetKernelArg(k, a++, sizeof(int), &p->img_h),
              "clSetKernelArg(conv3x3)");
    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, 2, NULL, global, local,
                                     0, NULL, NULL),
              "clEnqueueNDRangeKernel(conv3x3)");
}

void dsen2cr_model_run(struct dsen2cr_model *m, const float *in, float *out,
                       int x0, int y0, int img_w, int img_h)
{
    struct ocl_backend *ocl = m->ocl;
    size_t plane = (size_t)m->tile_w * m->tile_h;
    struct tile_pos p = {x0, y0, img_w, img_h};
    cl_mem x = m->a, t = m->b, y = m->c, swap;
    int i;

    ocl_write(ocl, m->in, 0, in, sizeof(float) * plane * m->nin);

    conv(m, &m->head, m->in, x, EP_RELU, 1.0f, NULL, &p);
    for (i = 0; i < m->nblocks; i++) {
        conv(m, &m->conv[2 * i], x, t, EP_RELU, 1.0f, NULL, &p);
        conv(m, &m->conv[2 * i + 1], t, y, EP_RESIDUAL, DSEN2CR_RES_SCALE, x,
             &p);
        swap = x;
        x = y;
        y = swap;
    }
    /* Long skip: the optical bands are the first S2_NBANDS input planes. */
    conv(m, &m->tail, x, m->out, EP_RESIDUAL, 1.0f, m->in, &p);

    ocl_read(ocl, m->out, 0, out, sizeof(float) * plane * S2_NBANDS);
}
