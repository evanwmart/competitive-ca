CC      = cc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra -pedantic

# fixed-lattice binary
OBJS     = sim.o stats.o main.o
torus: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

# dynamic-graph binary
OBJS_DYN = sim.o dgraph.o main_dyn.o
torus_dyn: $(OBJS_DYN)
	$(CC) $(CFLAGS) $(OBJS_DYN) -o $@

all: torus torus_dyn

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sim.o:      sim.c sim.h
stats.o:    stats.c stats.h sim.h
main.o:     main.c sim.h stats.h
dgraph.o:   dgraph.c dgraph.h sim.h
main_dyn.o: main_dyn.c dgraph.h sim.h

clean:
	rm -f torus torus_dyn $(OBJS) $(OBJS_DYN)

.PHONY: all clean
