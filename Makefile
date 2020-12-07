MPICC ?= mpicc
MPICXX ?= mpicxx
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra

sources = $(wildcard *.cc)
headers = $(wildcard *.hh)
executables = $(sources:.cc=)

all: $(executables)

%: %.cc $(headers)
	$(MPICXX) -o $@ $(CXXFLAGS) $<

clean:
	rm -rf $(executables)
