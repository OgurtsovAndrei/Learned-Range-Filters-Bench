.PHONY: build-filters test clean

build-filters:
	$(MAKE) -C grafite build
	$(MAKE) -C surf build
	$(MAKE) -C snarf build

test: build-filters
	cd bench && go test -v -timeout 60m .

clean:
	$(MAKE) -C grafite clean
	$(MAKE) -C surf clean
	$(MAKE) -C snarf clean
