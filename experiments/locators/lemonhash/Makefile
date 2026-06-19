# Makefile for LeMonHash CGO Module

# Detect OS for specific flags if needed
OS := $(shell uname -s)
BREW_PREFIX := $(shell [ -d /opt/homebrew ] && echo /opt/homebrew || echo /usr/local)

.PHONY: all build clean test bench help

all: build test

help:
	@echo "Available targets:"
	@echo "  build   - Build the C++ static libraries using CMake"
	@echo "  test    - Run Go tests (builds C++ first if needed)"
	@echo "  bench   - Run Go benchmarks"
	@echo "  clean   - Remove build artifacts"

build:
	mkdir -p build
	cd build && cmake -DCMAKE_CXX_FLAGS="-I$(BREW_PREFIX)/include" \
		-DSUCCINCT_USE_INTRINSICS=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_BUILD_TYPE=Release ..
	cd build && $(MAKE)

test: build
	go test -v .

bench: build
	go test -v -bench=. -benchmem .

clean:
	rm -rf build
