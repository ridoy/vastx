VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X github.com/ridoy/vastx/internal/cli.Version=$(VERSION)
PREFIX  ?= $(HOME)/.local
GOFLAGS := -trimpath

.PHONY: build install test lint clean

# Builds the host binary plus static Linux binaries. `vastx share` uploads the
# Linux binary sitting next to the host one so instances can bootstrap.
build:
	go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o bin/vastx ./cmd/vastx
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o bin/vastx-linux-amd64 ./cmd/vastx
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o bin/vastx-linux-arm64 ./cmd/vastx

install: build
	mkdir -p $(PREFIX)/bin
	cp bin/vastx bin/vastx-linux-amd64 bin/vastx-linux-arm64 $(PREFIX)/bin/

test:
	go test -race ./...

lint:
	go vet ./...
	test -z "$$(gofmt -l .)" || (gofmt -l . && exit 1)

clean:
	rm -rf bin
