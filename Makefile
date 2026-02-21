# Flash Attention CUDA Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 --use_fast_math
ARCH = sm_86  # RTX 3080 Ti - change for your GPU

TARGET = flash_attention
SRC = flash_attention.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
