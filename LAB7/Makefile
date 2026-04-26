# Compiler
NVCC = nvcc
NVCCFLAGS = -O2 -arch=native -lineinfo

# Exercises
EXERCISES = exercises/q1_thread_tasks exercises/q2_merge_sort exercises/q3_vector_profiling

ALL = $(EXERCISES)

.PHONY: all clean run-q1 run-q2 run-q3 run-exercises

all: $(ALL)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

run-q1: exercises/q1_thread_tasks
	@echo "\n=== Problem 1: Thread Tasks ==="
	exercises/q1_thread_tasks

run-q2: exercises/q2_merge_sort
	@echo "\n=== Problem 2: Merge Sort ==="
	exercises/q2_merge_sort

run-q3: exercises/q3_vector_profiling
	@echo "\n=== Problem 3: Vector Profiling ==="
	exercises/q3_vector_profiling

run-exercises: run-q1 run-q2 run-q3

clean:
	rm -f $(ALL)
