MODULE_TOPDIR = ../..

PGM = i.dsen2cr.opencl

LIBES = $(RASTERLIB) $(GISLIB) $(MATHLIB)
DEPENDENCIES = $(RASTERDEP) $(GISDEP)
EXTRA_INC = $(OCLINCPATH) -I$(OBJDIR)
EXTRA_CFLAGS = -O3 -std=gnu11
EXTRA_LIBS = $(OCLLIBPATH) $(OCLLIB)

# The checkpoint converter is installed next to the module data.
ETCFILES = dsen2cr_convert.py

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# Embed the OpenCL kernels as a C string literal.
$(OBJDIR)/ocl_backend.o: $(OBJDIR)/dsen2cr_kernels_cl.h

$(OBJDIR)/dsen2cr_kernels_cl.h: dsen2cr_kernels.cl | $(OBJDIR)
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@
