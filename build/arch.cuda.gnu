LIB_BLAS  = -lblas -lpthread -lm
LDLIBS   += $(LIB_BLAS)
INCLUDES += -I/usr/include/openblas $(shell pkg-config --cflags-only-I openblas 2>/dev/null)
C++FLAGS += $(INCLUDES)

CUDA_INSTALL_PATH=/usr/local/cuda-11.6

# Target GPU: 75 = T4 (default). Override for other GPUs, e.g. `make SM=121` on a GB10.
SM ?= 75
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
PTXFLAGS=-v
# PTXFLAGS=-dlcm=ca 
NVCCFLAGS= -O3 $(GENCODE_FLAGS) -c

# Compilers
NVCC            = $(shell which nvcc)
C++             = $(shell which g++)
C++LINK         = $(C++)
NVCCLINK        = $(NVCC)
CLINK           = $(CC)

.SUFFIXES:
.SUFFIXES: .cpp .c .cu .o

.cpp.o:
		$(C++) $(C++FLAGS) -c $<

.c.o:
		$(C++) $(C++FLAGS) -c $<

.cu.o:
	$(NVCC)  $(NVCCFLAGS) $<


