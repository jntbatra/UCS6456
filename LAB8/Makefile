# ============================================================
# Makefile — CUDA DIY Exercises 1-5
# ============================================================
# Usage:
#   make all        — build all 5 exercises
#   make ex01       — build only Exercise 1
#   make clean      — remove all binaries
#   make run_ex01   — build and run Exercise 1
#
# Adjust SM (compute capability) to match your GPU:
#   RTX 2060/2070/2080 : sm_75
#   RTX 3060/3070/3080 : sm_86
#   RTX 4080/4090      : sm_89
#   A100               : sm_80
#   H100               : sm_90
# ============================================================

SM        := native
NVCC      := nvcc
NVCCFLAGS := -O2 -arch=$(SM) -lineinfo

# cuDNN paths (pip install nvidia-cudnn-cu12, or set CUDNN_PATH)
CUDNN_PATH ?= $(shell python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)
CUDNN_INC  := $(if $(CUDNN_PATH),-I$(CUDNN_PATH)/include)
CUDNN_LIB  := $(if $(CUDNN_PATH),-L$(CUDNN_PATH)/lib -Xlinker -rpath -Xlinker $(CUDNN_PATH)/lib)

# Libraries
LIBS_BASE  := -lm
LIBS_BLAS  := -lcublas $(LIBS_BASE)
LIBS_FULL  := $(CUDNN_LIB) -lcudnn -lcublas $(LIBS_BASE)

.PHONY: all clean ex01 ex02 ex03 ex04 ex05 \
        run_ex01 run_ex02 run_ex03 run_ex04

all: ex01 ex02 ex03 ex04 ex05

ex01: ex01_cuda_basics.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex02: ex02_memory_hierarchy.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex03: ex03_ml_primitives.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex04: ex04_cnn_layers.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BLAS)
	@echo "[✓] Built $@"

ex05: ex05_mnist_cnn.cu
	$(NVCC) $(NVCCFLAGS) $(CUDNN_INC) $< -o $@ $(LIBS_FULL)
	@echo "[✓] Built $@"

run_ex01: ex01
	./ex01

run_ex02: ex02
	./ex02

run_ex03: ex03
	./ex03

run_ex04: ex04
	./ex04

clean:
	rm -f ex01 ex02 ex03 ex04 ex05
	@echo "[✓] Cleaned"
