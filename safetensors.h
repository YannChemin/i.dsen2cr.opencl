#ifndef I_DSEN2CR_OPENCL_SAFETENSORS_H
#define I_DSEN2CR_OPENCL_SAFETENSORS_H

#include <stddef.h>

#define ST_MAX_DIMS 8

struct st_tensor {
    char *name;
    char dtype[8];
    int ndim;
    long shape[ST_MAX_DIMS];
    size_t begin, end; /* Byte range in the data section. */
};

struct st_file {
    char *path;
    int n;
    struct st_tensor *t;
    unsigned char *data; /* Data section, after the JSON header. */
    size_t data_size;
};

/* Read a safetensors file entirely into memory. Fatal on any format
 * error. */
void st_open(struct st_file *f, const char *path);
void st_close(struct st_file *f);

/* Tensor lookup by name, NULL when absent. */
const struct st_tensor *st_find(const struct st_file *f, const char *name);

/* Return a newly allocated float32 copy of the named tensor, converting
 * from F16/BF16/F64 where needed. When ndim > 0, the tensor must have
 * exactly that shape (given in shape[]; -1 matches any extent) and the
 * actual shape is written back into shape[]. Fatal when the tensor is
 * missing or its shape differs. */
float *st_get_f32(const struct st_file *f, const char *name, int ndim,
                  long *shape);

#endif /* I_DSEN2CR_OPENCL_SAFETENSORS_H */
