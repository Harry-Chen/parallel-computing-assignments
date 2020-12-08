include ./Makefrag

sources = $(wildcard *.cc)
headers = $(wildcard *.hh)
executables = $(sources:.cc=)

all: $(executables)

%: %.cc $(headers)
	$(MPICXX) -o $@ $(CXXFLAGS) $<

clean:
	rm -rf $(executables)
