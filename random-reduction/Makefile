include ../Makefrag

sources = $(wildcard *.cpp)
headers = $(wildcard *.hpp)
executables = $(sources:.cpp=)

all: $(executables)

%: %.cpp $(headers)
	$(MPICXX) -o $@ $(CXXFLAGS) -O3 -march=native $<

clean:
	rm -rf $(executables)
