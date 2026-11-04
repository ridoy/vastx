package cli

import (
	"bytes"
	"context"
	"crypto/rand"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/johannesboyne/gofakes3"
	"github.com/johannesboyne/gofakes3/backend/s3mem"
)

const testBucket = "models"

type fakeS3 struct {
	url    string
	ranged atomic.Int64 // ranged GETs served
}

// startFakeS3 runs an in-memory S3 server and points vastx's config at it.
func startFakeS3(t *testing.T) *fakeS3 {
	t.Helper()
	backend := s3mem.New()
	if err := backend.CreateBucket(testBucket); err != nil {
		t.Fatal(err)
	}
	f := &fakeS3{}
	h := gofakes3.New(backend).Server()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.Header.Get("Range") != "" {
			f.ranged.Add(1)
		}
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(ts.Close)
	f.url = ts.URL
	t.Setenv("VASTX_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	t.Setenv("VASTX_S3_ENDPOINT", ts.URL)
	t.Setenv("VASTX_S3_REGION", "us-east-1")
	t.Setenv("VASTX_R2_ACCESS_KEY_ID", "test")
	t.Setenv("VASTX_R2_SECRET_ACCESS_KEY", "test-secret")
	t.Setenv("VASTX_R2_BUCKET", testBucket)
	return f
}

func vastx(t *testing.T, args ...string) (stdout, stderr string) {
	t.Helper()
	out, errOut, err := vastxErr(args...)
	if err != nil {
		t.Fatalf("vastx %s: %v\nstdout:\n%s\nstderr:\n%s", strings.Join(args, " "), err, out, errOut)
	}
	return out, errOut
}

func vastxErr(args ...string) (stdout, stderr string, err error) {
	var o, e bytes.Buffer
	cmd := NewRootCmd()
	cmd.SetArgs(args)
	cmd.SetOut(&o)
	cmd.SetErr(&e)
	cmd.SetIn(strings.NewReader(""))
	err = cmd.ExecuteContext(context.Background())
	return o.String(), e.String(), err
}

func writeRandom(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	b := make([]byte, size)
	if _, err := rand.Read(b); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// readTree maps relative path to content for every regular file under dir,
// following symlinks.
func readTree(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	var walk func(abs, rel string)
	walk = func(abs, rel string) {
		entries, err := os.ReadDir(abs)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			p := filepath.Join(abs, e.Name())
			r := filepath.ToSlash(filepath.Join(rel, e.Name()))
			fi, err := os.Stat(p)
			if err != nil {
				t.Fatal(err)
			}
			if fi.IsDir() {
				walk(p, r)
				continue
			}
			b, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			out[r] = b
		}
	}
	walk(dir, "")
	return out
}

func assertTree(t *testing.T, got, want map[string][]byte) {
	t.Helper()
	for p, w := range want {
		g, ok := got[p]
		if !ok {
			t.Errorf("missing %s", p)
			continue
		}
		if !bytes.Equal(g, w) {
			t.Errorf("%s: content differs (got %d bytes, want %d)", p, len(g), len(w))
		}
	}
	for p := range got {
		if _, ok := want[p]; !ok {
			t.Errorf("unexpected file %s", p)
		}
	}
}

func TestPushPullShareEndToEnd(t *testing.T) {
	s3 := startFakeS3(t)
	const mib = 1 << 20

	// A model directory shaped like a Hugging Face snapshot: shards spanning
	// several chunks, small JSON files, a nested dir, an empty file, a
	// symlinked file, and metadata that must not be uploaded.
	src := filepath.Join(t.TempDir(), "tiny-llama")
	writeRandom(t, filepath.Join(src, "model-00001-of-00002.safetensors"), 12*mib+123)
	writeRandom(t, filepath.Join(src, "model-00002-of-00002.safetensors"), 5*mib)
	writeRandom(t, filepath.Join(src, "config.json"), 700)
	writeRandom(t, filepath.Join(src, "tokenizer", "tokenizer.model"), 3*mib)
	if err := os.WriteFile(filepath.Join(src, "empty.txt"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	writeRandom(t, filepath.Join(src, ".git", "HEAD"), 10)
	writeRandom(t, filepath.Join(src, ".cache", "huggingface", "download", "x.lock"), 10)
	blob := filepath.Join(t.TempDir(), "blob")
	writeRandom(t, blob, 2*mib+1)
	if err := os.Symlink(blob, filepath.Join(src, "linked.bin")); err != nil {
		t.Fatal(err)
	}

	want := readTree(t, src)
	for p := range want {
		if strings.HasPrefix(p, ".git/") || strings.HasPrefix(p, ".cache/") {
			delete(want, p)
		}
	}

	// Push. 5 MiB chunks put the first shard in three multipart parts.
	out, _ := vastx(t, "push", "-q", "--chunk-size", "5MiB", src, "test/tiny-llama")
	if !strings.Contains(out, "pushed test/tiny-llama: 6 files") {
		t.Fatalf("unexpected push output: %s", out)
	}

	out, _ = vastx(t, "ls")
	if !strings.Contains(out, "test/tiny-llama") || !strings.Contains(out, " 6 ") {
		t.Fatalf("ls doesn't list the model: %s", out)
	}

	// Pull by name with credentials.
	dst := filepath.Join(t.TempDir(), "byname")
	vastx(t, "pull", "-q", "test/tiny-llama", dst)
	assertTree(t, readTree(t, dst), want)
	if s3.ranged.Load() == 0 {
		t.Fatal("pull didn't use ranged requests")
	}

	// Pushing again uploads nothing.
	out, _ = vastx(t, "push", "-q", "--chunk-size", "5MiB", src, "test/tiny-llama")
	if !strings.Contains(out, "uploaded 0 files") || !strings.Contains(out, "6 unchanged") {
		t.Fatalf("re-push wasn't a no-op: %s", out)
	}

	// Corruption on disk and an interrupted file are repaired by pulling again,
	// fetching only what's wrong.
	shard := filepath.Join(dst, "model-00001-of-00002.safetensors")
	fh, err := os.OpenFile(shard, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fh.WriteAt([]byte("garbage"), 6*mib); err != nil {
		t.Fatal(err)
	}
	fh.Close()
	tok := filepath.Join(dst, "tokenizer", "tokenizer.model")
	if err := os.Rename(tok, tok+".vastx-partial"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tok+".vastx-partial", make([]byte, 3*mib), 0o644); err != nil {
		t.Fatal(err)
	}
	out, _ = vastx(t, "pull", "-q", "test/tiny-llama", dst)
	assertTree(t, readTree(t, dst), want)
	// One 5 MiB chunk of the shard plus the 3 MiB tokenizer.
	if !strings.Contains(out, "downloaded 8.0 MiB") {
		t.Fatalf("repair pull should download only the bad chunks: %s", out)
	}

	// Share link: pull with no credentials at all.
	bin := filepath.Join(t.TempDir(), "vastx-linux-amd64")
	writeRandom(t, bin, 1000)
	shareURL, _ := vastx(t, "share", "test/tiny-llama", "--output", "url", "--no-binary")
	shareURL = strings.TrimSpace(shareURL)
	cmdLine, _ := vastx(t, "share", "test/tiny-llama", "--output", "cmd", "--binary", bin, "--dest", "/workspace/m")
	if !strings.HasPrefix(cmdLine, "curl -fsSL -o /usr/local/bin/vastx '") || !strings.Contains(cmdLine, "vastx pull '") ||
		!strings.HasSuffix(strings.TrimSpace(cmdLine), "'/workspace/m'") {
		t.Fatalf("unexpected bootstrap command: %s", cmdLine)
	}
	binURL := strings.Split(strings.Split(cmdLine, "'")[1], "'")[0]
	resp, err := http.Get(binURL)
	if err != nil {
		t.Fatal(err)
	}
	gotBin, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	wantBin, _ := os.ReadFile(bin)
	if !bytes.Equal(gotBin, wantBin) {
		t.Fatal("bootstrap binary URL doesn't serve the uploaded binary")
	}

	for _, k := range []string{"VASTX_R2_ACCESS_KEY_ID", "VASTX_R2_SECRET_ACCESS_KEY", "VASTX_R2_BUCKET", "VASTX_S3_ENDPOINT"} {
		t.Setenv(k, "")
	}
	dst2 := filepath.Join(t.TempDir(), "byurl")
	vastx(t, "pull", "-q", shareURL, dst2)
	assertTree(t, readTree(t, dst2), want)
	if _, _, err := vastxErr("ls"); err == nil || !strings.Contains(err.Error(), "missing storage config") {
		t.Fatalf("ls without credentials should fail with a config error, got %v", err)
	}
	t.Setenv("VASTX_R2_ACCESS_KEY_ID", "test")
	t.Setenv("VASTX_R2_SECRET_ACCESS_KEY", "test-secret")
	t.Setenv("VASTX_R2_BUCKET", testBucket)
	t.Setenv("VASTX_S3_ENDPOINT", s3.url)

	// Files removed locally are removed from the bucket on the next push.
	if err := os.RemoveAll(filepath.Join(src, "tokenizer")); err != nil {
		t.Fatal(err)
	}
	delete(want, "tokenizer/tokenizer.model")
	out, _ = vastx(t, "push", "-q", "--chunk-size", "5MiB", src, "test/tiny-llama")
	if !strings.Contains(out, "removed 1 file deleted locally") {
		t.Fatalf("push should remove the deleted file: %s", out)
	}
	dst3 := filepath.Join(t.TempDir(), "afterdelete")
	vastx(t, "pull", "-q", "test/tiny-llama", dst3)
	assertTree(t, readTree(t, dst3), want)

	// Names can't nest inside each other.
	if _, _, err := vastxErr("push", "-q", src, "test/tiny-llama/nested"); err == nil {
		t.Fatal("pushing a model nested inside another should fail")
	}
	if _, _, err := vastxErr("push", "-q", src, "test"); err == nil {
		t.Fatal("pushing a model that would contain another should fail")
	}

	// rm needs confirmation, then deletes everything.
	if _, _, err := vastxErr("rm", "test/tiny-llama"); err == nil {
		t.Fatal("rm without --yes and no input should fail")
	}
	vastx(t, "rm", "--yes", "test/tiny-llama")
	_, errOut := vastx(t, "ls")
	if !strings.Contains(errOut, "no models") {
		t.Fatalf("model still listed after rm: %s", errOut)
	}
	if _, _, err := vastxErr("pull", "-q", "test/tiny-llama", t.TempDir()); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("pull of a deleted model should say not found, got %v", err)
	}
}

func TestPullSingleFile(t *testing.T) {
	startFakeS3(t)
	f := filepath.Join(t.TempDir(), "weights.gguf")
	writeRandom(t, f, 7<<20)
	vastx(t, "push", "-q", "--chunk-size", "5MiB", f)
	dst := t.TempDir()
	vastx(t, "pull", "-q", "weights.gguf", dst)
	want, _ := os.ReadFile(f)
	got, err := os.ReadFile(filepath.Join(dst, "weights.gguf"))
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("single file round trip failed: %v", err)
	}
	// No partial files left behind.
	_ = filepath.WalkDir(dst, func(p string, d fs.DirEntry, _ error) error {
		if strings.HasSuffix(p, ".vastx-partial") {
			t.Errorf("leftover partial file %s", p)
		}
		return nil
	})
}
