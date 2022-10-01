# uncomment to have debug build
//DEBUG ?= 1

SRC_DIR := ../../src
SRC_MSH_DIR := ../../src/mesh
VPATH := $(SRC_DIR):$(SRC_MSH_DIR)

# CUDA root path
CUDA_PATH := /usr/local/cuda-10.2

# GPU
GPU_CARD := -arch=sm_72 # specify the proper device compute capability here

# =========== CUDA part ===========
NVCC := $(CUDA_PATH)/bin/nvcc
# separate compilation -- the above CUDA version supports c++ up to 14
NVCC_FLAGS := -std=c++14 -dc -Xcompiler
ifeq ($(DEBUG),1)
  NVCC_FLAGS += -g3 -O0 -lineinfo
else
  NVCC_FLAGS += -O3
endif
CUDA_LINK_FLAGS := -dlink

CUDA_INCL := -I$(CUDA_PATH)/include
CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcusparse -lcusolver

CUDA_SRCS := $(wildcard *.cu) $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS := $(patsubst %.cu, %.o, $(notdir $(CUDA_SRCS)))

# =========== C++ part ===========
CC := g++
CFLAGS := -O3 -std=c++17 -Wall
ifeq ($(DEBUG),1)
  CFLAGS += -g3 -O0
endif

INCL := -I$(SRC_DIR) -I$(SRC_MSH_DIR)
LIBS :=# ARE THERE LICENSE ISSUES OF USING THESE LIBRARIES?

SRCS := $(wildcard *.cpp) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_MSH_DIR)/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(notdir $(SRCS)))

# =========== build  ===========
EXEC := stokes_2d
CUDA_LINK_OBJ := cuLink.o

all: $(EXEC)
$(EXEC): $(CUDA_OBJS) $(OBJS)
	$(NVCC) $(GPU_CARD) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS) $(CUDA_OBJS) $(CUDA_LINK_OBJ) $(CUDA_LIBS)

%.o: %.cpp
	$(CC) $(INCL) $(CUDA_INCL) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) $(INCL) $(CUDA_INCL) -c $< -o $@

clean:	
	rm -f $(OBJS) $(EXEC) *.o
	
.PHONY : all clean
