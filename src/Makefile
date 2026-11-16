# Build the C++ Track 3 entry: make -C src   ->  src/pathfinder
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
pathfinder: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp -lz
clean:
	rm -f pathfinder
.PHONY: clean
