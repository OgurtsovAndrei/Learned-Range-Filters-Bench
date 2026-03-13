OS := $(shell uname -s)
BREW_PREFIX := $(shell [ -d /opt/homebrew ] && echo /opt/homebrew || echo /usr/local)

.PHONY: all build clean

all: build

build:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_FLAGS="-I$(BREW_PREFIX)/include" \
		-DBUILD_SHARED_LIBS=OFF ..
	cd build && $(MAKE)

clean:
	rm -rf build
