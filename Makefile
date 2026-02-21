# Flash Attention CUDA Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 --use_fast_math
ARCH = sm_86  # RTX 3080 Ti - change for your GPU

TARGET = flash_attention
SRC = flash_attention.cu

TEST_TARGET = test_kernel
TEST_SRC = cuda/test_kernel.cu

.PHONY: all clean run test run_test

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) $< -o $@

test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) $< -o $@

clean:
	rm -f $(TARGET) $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

run_test: $(TEST_TARGET)
	./$(TEST_TARGET)
