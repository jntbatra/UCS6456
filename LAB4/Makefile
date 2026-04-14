# Compiler
MPICC = /usr/lib64/openmpi/bin/mpicc
MPIRUN = /usr/lib64/openmpi/bin/mpirun

# Flags
CFLAGS = -Wall -O3 -lm

# Examples
EXAMPLES = examples/hello_mpi examples/send_recv examples/broadcast examples/reduce_sum examples/matrix_mult

# Exercises
EXERCISES = exercises/q1_daxpy exercises/q2_broadcast_race exercises/q3_dot_product exercises/q4_primes exercises/q5_perfect_numbers

# All targets
ALL = $(EXAMPLES) $(EXERCISES)

.PHONY: all clean run-examples run-exercises run-q1 run-q2 run-q3 run-q4 run-q5

all: $(ALL)

# Pattern rule for all C files
%: %.c
	$(MPICC) $(CFLAGS) -o $@ $<

# Run examples
run-examples: $(EXAMPLES)
	@echo "=== Hello MPI ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/hello_mpi
	@echo "\n=== Send/Recv ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 2 examples/send_recv
	@echo "\n=== Broadcast ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/broadcast
	@echo "\n=== Reduce Sum ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/reduce_sum
	@echo "\n=== Matrix Mult ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/matrix_mult

# Run individual exercises
run-q1: exercises/q1_daxpy
	@for np in 1 2 4 8; do \
		echo "\n--- Q1 DAXPY: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q1_daxpy; \
	done

run-q2: exercises/q2_broadcast_race
	@for np in 2 4 8 16; do \
		echo "\n--- Q2 Broadcast Race: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q2_broadcast_race; \
	done

run-q3: exercises/q3_dot_product
	@for np in 1 2 4 8; do \
		echo "\n--- Q3 Dot Product: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q3_dot_product; \
	done

run-q4: exercises/q4_primes
	@for np in 2 4 8; do \
		echo "\n--- Q4 Primes: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q4_primes; \
	done

run-q5: exercises/q5_perfect_numbers
	@for np in 2 4 8; do \
		echo "\n--- Q5 Perfect Numbers: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q5_perfect_numbers; \
	done

run-exercises: run-q1 run-q2 run-q3 run-q4 run-q5

clean:
	rm -f $(ALL)
