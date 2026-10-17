/****************************************************************************
 *
 * MODULE:       i.dsen2cr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Minimal reader for the safetensors weight format written by
 *               dsen2cr_convert.py.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "safetensors.h"

/* The header is a flat JSON object mapping tensor names to
 * {"dtype": ..., "shape": [...], "data_offsets": [begin, end]}, plus an
 * optional "__metadata__" object of strings. This parser handles exactly
 * that subset and fails on anything else. */
struct parser {
    const char *s, *end, *path;
};

static void perr(const struct parser *p, const char *what)
{
    G_fatal_error(_("<%s> is not a valid safetensors file: %s at header "
                    "offset %ld"),
                  p->path, what, (long)(p->end - p->s));
}

static void skip_ws(struct parser *p)
{
    while (p->s < p->end && isspace((unsigned char)*p->s))
        p->s++;
}

static void expect(struct parser *p, char c)
{
    skip_ws(p);
    if (p->s >= p->end || *p->s != c) {
        char msg[32];

        snprintf(msg, sizeof(msg), "expected '%c'", c);
        perr(p, msg);
    }
    p->s++;
}

static int peek(struct parser *p, char c)
{
    skip_ws(p);
    return p->s < p->end && *p->s == c;
}

/* Parse a JSON string. Tensor names and dtypes are plain ASCII, so escape
 * sequences are kept verbatim except for \" and \\. */
static char *parse_string(struct parser *p)
{
    const char *start;
    char *out;
    size_t n = 0;

    expect(p, '"');
    start = p->s;
    while (p->s < p->end && *p->s != '"') {
        if (*p->s == '\\')
            p->s++;
        p->s++;
    }
    if (p->s >= p->end)
        perr(p, "unterminated string");
    out = G_malloc(p->s - start + 1);
    while (start < p->s) {
        if (*start == '\\' && (start[1] == '"' || start[1] == '\\'))
            start++;
        out[n++] = *start++;
    }
    out[n] = '\0';
    p->s++;
    return out;
}

static long parse_int(struct parser *p)
{
    char *e;
    long v;

    skip_ws(p);
    v = strtol(p->s, &e, 10);
    if (e == p->s || e > p->end)
        perr(p, "expected an integer");
    p->s = e;
    return v;
}

static int parse_int_array(struct parser *p, long *v, int max)
{
    int n = 0;

    expect(p, '[');
    if (peek(p, ']')) {
        p->s++;
        return 0;
    }
    for (;;) {
        long x = parse_int(p);

        if (n == max)
            perr(p, "too many array elements");
        v[n++] = x;
        if (peek(p, ',')) {
            p->s++;
            continue;
        }
        expect(p, ']');
        return n;
    }
}

/* Skip the __metadata__ object, whose values are all strings. */
static void skip_string_object(struct parser *p)
{
    expect(p, '{');
    if (peek(p, '}')) {
        p->s++;
        return;
    }
    for (;;) {
        G_free(parse_string(p));
        expect(p, ':');
        G_free(parse_string(p));
        if (peek(p, ',')) {
            p->s++;
            continue;
        }
        expect(p, '}');
        return;
    }
}

static void parse_tensor(struct parser *p, struct st_tensor *t)
{
    long offs[2];
    int have = 0;

    expect(p, '{');
    for (;;) {
        char *key = parse_string(p);

        expect(p, ':');
        if (strcmp(key, "dtype") == 0) {
            char *v = parse_string(p);

            if (strlen(v) >= sizeof(t->dtype))
                perr(p, "unknown dtype");
            strcpy(t->dtype, v);
            G_free(v);
            have |= 1;
        }
        else if (strcmp(key, "shape") == 0) {
            t->ndim = parse_int_array(p, t->shape, ST_MAX_DIMS);
            have |= 2;
        }
        else if (strcmp(key, "data_offsets") == 0) {
            if (parse_int_array(p, offs, 2) != 2 || offs[0] < 0 ||
                offs[1] < offs[0])
                perr(p, "bad data_offsets");
            t->begin = offs[0];
            t->end = offs[1];
            have |= 4;
        }
        else
            perr(p, "unexpected tensor field");
        G_free(key);
        if (peek(p, ',')) {
            p->s++;
            continue;
        }
        expect(p, '}');
        break;
    }
    if (have != 7)
        perr(p, "incomplete tensor entry");
}

void st_open(struct st_file *f, const char *path)
{
    struct parser p;
    unsigned char len8[8];
    uint64_t hlen = 0;
    long fsize;
    char *header;
    FILE *fp;
    int cap = 64, i;

    memset(f, 0, sizeof(*f));
    f->path = G_store(path);
    fp = fopen(path, "rb");
    if (!fp)
        G_fatal_error(_("Unable to open model file <%s>"), path);
    fseek(fp, 0, SEEK_END);
    fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize < 8 || fread(len8, 1, 8, fp) != 8)
        G_fatal_error(_("<%s> is not a valid safetensors file: too short"),
                      path);
    for (i = 7; i >= 0; i--)
        hlen = (hlen << 8) | len8[i];
    if (hlen < 2 || hlen > (uint64_t)fsize - 8)
        G_fatal_error(_("<%s> is not a valid safetensors file: bad header "
                        "length"),
                      path);

    header = G_malloc(hlen);
    f->data_size = fsize - 8 - hlen;
    f->data = G_malloc(f->data_size ? f->data_size : 1);
    if (fread(header, 1, hlen, fp) != hlen ||
        fread(f->data, 1, f->data_size, fp) != f->data_size)
        G_fatal_error(_("Error reading model file <%s>"), path);
    fclose(fp);

    p.s = header;
    p.end = header + hlen;
    p.path = path;
    f->t = G_malloc(cap * sizeof(*f->t));
    expect(&p, '{');
    if (!peek(&p, '}')) {
        for (;;) {
            char *name = parse_string(&p);

            expect(&p, ':');
            if (strcmp(name, "__metadata__") == 0) {
                skip_string_object(&p);
                G_free(name);
            }
            else {
                struct st_tensor *t;

                if (f->n == cap) {
                    cap *= 2;
                    f->t = G_realloc(f->t, cap * sizeof(*f->t));
                }
                t = &f->t[f->n++];
                memset(t, 0, sizeof(*t));
                t->name = name;
                parse_tensor(&p, t);
                if (t->end > f->data_size)
                    perr(&p, "tensor data beyond end of file");
            }
            if (peek(&p, ',')) {
                p.s++;
                continue;
            }
            break;
        }
    }
    expect(&p, '}');
    G_free(header);
    G_debug(1, "safetensors <%s>: %d tensors", path, f->n);
}

void st_close(struct st_file *f)
{
    int i;

    for (i = 0; i < f->n; i++)
        G_free(f->t[i].name);
    G_free(f->t);
    G_free(f->data);
    G_free(f->path);
    memset(f, 0, sizeof(*f));
}

const struct st_tensor *st_find(const struct st_file *f, const char *name)
{
    int i;

    for (i = 0; i < f->n; i++)
        if (strcmp(f->t[i].name, name) == 0)
            return &f->t[i];
    return NULL;
}

static float half_to_float(uint16_t h)
{
    int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;

    if (e == 0)
        v = ldexpf((float)m, -24);
    else if (e == 31)
        v = m ? NAN : INFINITY;
    else
        v = ldexpf((float)(m | 0x400), e - 25);
    return (h & 0x8000) ? -v : v;
}

float *st_get_f32(const struct st_file *f, const char *name, int ndim,
                  long *shape)
{
    const struct st_tensor *t = st_find(f, name);
    const unsigned char *src;
    size_t count = 1, i, esize;
    float *out;
    int d;

    if (!t)
        G_fatal_error(_("Tensor <%s> not found in model file <%s>"), name,
                      f->path);
    if (ndim > 0) {
        int ok = t->ndim == ndim;

        for (d = 0; ok && d < ndim; d++)
            if (shape[d] >= 0 && shape[d] != t->shape[d])
                ok = 0;
        if (!ok) {
            char got[128] = "", want[128] = "", buf[32];

            for (d = 0; d < t->ndim; d++) {
                snprintf(buf, sizeof(buf), d ? "x%ld" : "%ld", t->shape[d]);
                strcat(got, buf);
            }
            for (d = 0; d < ndim; d++) {
                if (shape[d] >= 0)
                    snprintf(buf, sizeof(buf), d ? "x%ld" : "%ld", shape[d]);
                else
                    snprintf(buf, sizeof(buf), d ? "x*" : "*");
                strcat(want, buf);
            }
            G_fatal_error(_("Tensor <%s> in <%s> has shape %s, expected %s"),
                          name, f->path, got, want);
        }
        for (d = 0; d < ndim; d++)
            shape[d] = t->shape[d];
    }
    for (d = 0; d < t->ndim; d++)
        count *= t->shape[d];

    if (strcmp(t->dtype, "F32") == 0)
        esize = 4;
    else if (strcmp(t->dtype, "F16") == 0 || strcmp(t->dtype, "BF16") == 0)
        esize = 2;
    else if (strcmp(t->dtype, "F64") == 0)
        esize = 8;
    else
        G_fatal_error(_("Tensor <%s> in <%s> has unsupported dtype %s"), name,
                      f->path, t->dtype);
    if (t->end - t->begin != count * esize)
        G_fatal_error(_("Tensor <%s> in <%s>: data size does not match its "
                        "shape"),
                      name, f->path);

    src = f->data + t->begin;
    out = G_malloc((count ? count : 1) * sizeof(float));
    for (i = 0; i < count; i++) {
        if (esize == 4)
            memcpy(&out[i], src + 4 * i, 4);
        else if (esize == 8) {
            double v;

            memcpy(&v, src + 8 * i, 8);
            out[i] = (float)v;
        }
        else {
            uint16_t h;

            memcpy(&h, src + 2 * i, 2);
            if (t->dtype[0] == 'B') {
                uint32_t u = (uint32_t)h << 16;

                memcpy(&out[i], &u, 4);
            }
            else
                out[i] = half_to_float(h);
        }
    }
    return out;
}
