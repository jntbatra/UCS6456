# Compiler
MPICC = /usr/lib64/openmpi/bin/mpicc
MPIRUN = /usr/lib64/openmpi/bin/mpirun

# Flags
CFLAGS = -Wall -O3 -lm

# Examples
EXAMPLES = examples/eg1_blocking_ring examples/eg2_nonblocking_ring examples/eg3_blocking_app examples/eg4_nonblocking_app examples/eg5_pi_calc

# Exercises
EXERCISES = exercises/q1_broadcast_race exercises/q2_blocking_vs_nonblocking exercises/q3_dot_product

ALL = $(EXAMPLES) $(EXERCISES)

.PHONY: all clean run-examples run-q1 run-q2 run-q3 run-exercises

all: $(ALL)

%: %.c
	$(MPICC) $(CFLAGS) -o $@ $<

run-examples: $(EXAMPLES)
	@echo "=== Eg1: Blocking Ring (small array — works) ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/eg1_blocking_ring
	@echo "\n=== Eg2: Non-Blocking Ring (10M — no deadlock) ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/eg2_nonblocking_ring
	@echo "\n=== Eg5: Pi Calculation ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 4 examples/eg5_pi_calc

run-blocking-apps: examples/eg3_blocking_app examples/eg4_nonblocking_app
	@echo "=== Eg3: Blocking App ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 2 examples/eg3_blocking_app
	@echo "\n=== Eg4: Non-Blocking App ===" && $(MPIRUN) --allow-run-as-root --oversubscribe -np 2 examples/eg4_nonblocking_app

run-q1: exercises/q1_broadcast_race
	@for np in 2 4 8 16; do \
		echo "\n--- Q1 Broadcast Race: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q1_broadcast_race; \
	done

run-q2: exercises/q2_blocking_vs_nonblocking
	@echo "--- Q2: Blocking vs Non-Blocking ---"
	$(MPIRUN) --allow-run-as-root --oversubscribe -np 2 exercises/q2_blocking_vs_nonblocking

run-q3: exercises/q3_dot_product
	@for np in 1 2 4 8; do \
		echo "\n--- Q3 Dot Product: $$np processes ---"; \
		$(MPIRUN) --allow-run-as-root --oversubscribe -np $$np exercises/q3_dot_product; \
	done

run-exercises: run-q1 run-q2 run-q3

clean:
	rm -f $(ALL)
