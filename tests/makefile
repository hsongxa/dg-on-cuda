DEBUG ?= 1

SRC_DIR := ../src
VPATH := $(SRC_DIR)

# CUDA root path
CUDA_PATH := /usr/local/cuda-10.2

# GPU
GPU_CARD := -arch=sm_72 # specify the proper device compute capability here

# =========== CUDA part ===========
NVCC := $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS := -O3 -dc -Xcompiler # separate compilation
ifeq ($(DEBUG),1)
  NVCC_FLAGS += -g3
endif

CUDA_LINK_FLAGS := -dlink

CUDA_INCL := -I$(CUDA_PATH)/include
CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcusparse

CUDA_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS := $(patsubst %.cu, %.o, $(notdir $(CUDA_SRCS)))

# =========== C++ part ===========
CC := g++
CFLAGS := -O3 -std=c++17 -Wall 
ifeq ($(DEBUG),1)
  CFLAGS += -g3
endif

INCL := -I$(SRC_DIR)
LIBS :=# -L~/gsl/lib -lgsl -lgslcblas -lm

SRCS := $(wildcard *.cpp) $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(notdir $(SRCS)))

# =========== build  ===========
EXEC := test
CUDA_LINK_OBJ := cuLink.o

all: $(EXEC)
$(EXEC): $(CUDA_OBJS) $(OBJS)
	$(NVCC) $(GPU_CARD) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS) $(CUDA_OBJS) $(CUDA_LINK_OBJ) $(CUDA_LIBS)

%.o: %.cpp
	$(CC) $(INCL) $(CUDA_INCL) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) $(CUDA_INCL) -c $< -o $@

clean:	
	rm -f $(OBJS) $(EXEC) *.o
	
.PHONY : all clean
