# Compiler
NVCC = nvcc
NVCCFLAGS = -O2 -arch=native -lineinfo

# Exercises
EXERCISES = exercises/a_device_query exercises/b_array_sum exercises/c_matrix_add

ALL = $(EXERCISES)

.PHONY: all clean run-a run-b run-c run-exercises

all: $(ALL)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

# Run individual exercises
run-a: exercises/a_device_query
	@echo "\n=== Part A: Device Query ==="
	exercises/a_device_query

run-b: exercises/b_array_sum
	@echo "\n=== Part B: Array Sum ==="
	exercises/b_array_sum

run-c: exercises/c_matrix_add
	@echo "\n=== Part C: Matrix Addition ==="
	exercises/c_matrix_add

run-exercises: run-a run-b run-c

clean:
	rm -f $(ALL)
