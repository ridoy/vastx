package transfer

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ridoy/vastx/internal/manifest"
)

func testManifest(t *testing.T, data []byte, chunk int64, url string) *manifest.Manifest {
	t.Helper()
	f := manifest.File{Path: "w.bin", Size: int64(len(data)), ChunkSize: chunk, URL: url}
	for i := range f.NumChunks() {
		off, n := f.ChunkRange(i)
		s := sha256.Sum256(data[off : off+n])
		f.Chunks = append(f.Chunks, hex.EncodeToString(s[:]))
	}
	return &manifest.Manifest{Version: manifest.FormatVersion, Name: "m", Files: []manifest.File{f}}
}

// A server that fails, corrupts and stalls requests must still produce a
// correct file. Each range fails three different ways before succeeding.
func TestPullRetriesFlakyServer(t *testing.T) {
	defer func(d time.Duration) { retryBaseDelay = d }(retryBaseDelay)
	retryBaseDelay = time.Millisecond

	data := make([]byte, 3<<20+17)
	rand.Read(data)
	var mu sync.Mutex
	attempts := map[string]int{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts[r.Header.Get("Range")]++
		n := attempts[r.Header.Get("Range")]
		mu.Unlock()
		switch n {
		case 1:
			http.Error(w, "slow down", http.StatusServiceUnavailable)
			return
		case 2:
			// Correct headers, corrupted body.
			bad := bytes.Clone(data)
			for i := range bad {
				bad[i] ^= 0xff
			}
			http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(bad))
			return
		case 3:
			// Send headers, then stall.
			w.Header().Set("Content-Length", "1048576")
			w.WriteHeader(http.StatusPartialContent)
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
			return
		}
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(data))
	}))
	defer ts.Close()

	dst := t.TempDir()
	m := testManifest(t, data, 1<<20, ts.URL)
	var log strings.Builder
	_, err := Pull(context.Background(), m, dst, PullOptions{
		Concurrency: 3, Verify: true, Log: &log, StallTimeout: 200 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("pull: %v\n%s", err, log.String())
	}
	got, _ := os.ReadFile(filepath.Join(dst, "w.bin"))
	if !bytes.Equal(got, data) {
		t.Fatal("content differs")
	}
	for _, want := range []string{"503", "checksum mismatch", "no data for"} {
		if !strings.Contains(log.String(), want) {
			t.Errorf("expected a retry logged for %q:\n%s", want, log.String())
		}
	}
}

func TestPullExpiredAndForbidden(t *testing.T) {
	data := []byte("hello")
	past := time.Now().Add(-time.Hour)
	m := testManifest(t, data, 1<<20, "http://127.0.0.1:1/x")
	m.ExpiresAt = &past
	if _, err := Pull(context.Background(), m, t.TempDir(), PullOptions{}); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("want expired error, got %v", err)
	}

	var hits atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, "<Error>Request has expired</Error>", http.StatusForbidden)
	}))
	defer ts.Close()
	m = testManifest(t, data, 1<<20, ts.URL)
	_, err := Pull(context.Background(), m, t.TempDir(), PullOptions{})
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Fatalf("want 403 error, got %v", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("403 should not be retried, got %d requests", hits.Load())
	}
}

func TestParseSize(t *testing.T) {
	cases := map[string]int64{"64MiB": 64 << 20, "64M": 64 << 20, "64mb": 64 << 20, "1g": 1 << 30, "512k": 512 << 10, "100": 100}
	for in, want := range cases {
		got, err := ParseSize(in)
		if err != nil || got != want {
			t.Errorf("ParseSize(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, in := range []string{"", "MiB", "12XB", "-5M"} {
		if _, err := ParseSize(in); err == nil {
			t.Errorf("ParseSize(%q) should fail", in)
		}
	}
}
