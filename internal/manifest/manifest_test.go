package manifest

import (
	"strings"
	"testing"
)

func TestChunkSizeFor(t *testing.T) {
	const mib = 1 << 20
	if got := ChunkSizeFor(100*mib, 64*mib); got != 64*mib {
		t.Errorf("small file: got %d", got)
	}
	// 1 TiB at 64 MiB would need 16384 parts, so the chunk size grows.
	got := ChunkSizeFor(1<<40, 64*mib)
	if got%mib != 0 || (1<<40+got-1)/got > MaxParts {
		t.Errorf("1 TiB: chunk size %d gives too many parts", got)
	}
}

func TestChunkRange(t *testing.T) {
	f := File{Size: 25, ChunkSize: 10}
	if f.NumChunks() != 3 {
		t.Fatalf("NumChunks = %d", f.NumChunks())
	}
	if off, n := f.ChunkRange(2); off != 20 || n != 5 {
		t.Errorf("last chunk = %d,%d", off, n)
	}
}

func TestValidPath(t *testing.T) {
	for _, p := range []string{"a", "a/b.json", "x/.hidden", "..a"} {
		if err := ValidPath(p); err != nil {
			t.Errorf("%q should be valid: %v", p, err)
		}
	}
	for _, p := range []string{"", "/etc/passwd", "../x", "a/../../x", "a//b", "a/", "./a", ".vastx/manifest.json", `a\b`} {
		if err := ValidPath(p); err == nil {
			t.Errorf("%q should be invalid", p)
		}
	}
}

func TestValidName(t *testing.T) {
	for _, n := range []string{"llama", "meta-llama/Llama-3.1-8B", "run_1/ckpt-2000"} {
		if err := ValidName(n); err != nil {
			t.Errorf("%q should be valid: %v", n, err)
		}
	}
	for _, n := range []string{"", "/a", "a/", "a//b", "..", "a/../b", ".vastx", "a/.vastx", "has space", "a?b"} {
		if err := ValidName(n); err == nil {
			t.Errorf("%q should be invalid", n)
		}
	}
}

func TestDecodeRejectsBadManifests(t *testing.T) {
	h := strings.Repeat("a", 64)
	cases := map[string]string{
		"version":   `{"version":2,"name":"m","files":[]}`,
		"traversal": `{"version":1,"name":"m","files":[{"path":"../x","size":1,"chunk_size":1,"chunks":["` + h + `"]}]}`,
		"chunks":    `{"version":1,"name":"m","files":[{"path":"x","size":20,"chunk_size":10,"chunks":["` + h + `"]}]}`,
		"duplicate": `{"version":1,"name":"m","files":[{"path":"x","size":0,"chunk_size":10,"chunks":[]},{"path":"x","size":0,"chunk_size":10,"chunks":[]}]}`,
		"file+dir":  `{"version":1,"name":"m","files":[{"path":"x","size":0,"chunk_size":10,"chunks":[]},{"path":"x/y","size":0,"chunk_size":10,"chunks":[]}]}`,
	}
	for name, js := range cases {
		if _, err := Decode(strings.NewReader(js)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}
