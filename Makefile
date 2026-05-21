.PHONY: test test-all vet clean tidy

# Run fast unit tests (default)
test:
	@echo "=== Running Fast Unit Tests ==="
	go test ./...

# Run all tests, including heavy benchmarks and dataset sweeps
test-all:
	@echo "=== Running All Tests (including heavy sweeps) ==="
	go test -tags=heavy ./...

# Vet and lint both root and submodule
vet:
	@echo "=== Vetting Root Code ==="
	go vet ./bench/...
	@echo "=== Vetting Submodule Code ==="
	cd Thesis && go vet ./...

# Run go mod tidy in both projects
tidy:
	@echo "=== Tidying Root go.mod ==="
	go mod tidy
	@echo "=== Tidying Submodule go.mod ==="
	cd Thesis && go mod tidy
