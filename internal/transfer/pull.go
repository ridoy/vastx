package transfer

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ridoy/vastx/internal/manifest"
)

const (
	// DefaultPullConcurrency is the number of ranged GETs in flight. Each
	// connection to R2 tops out well below a datacenter NIC, so pulls need
	// many of them.
	DefaultPullConcurrency = 32
	// DefaultStallTimeout aborts and retries a request that receives no data
	// for this long.
	DefaultStallTimeout = 60 * time.Second

	partialSuffix   = ".vastx-partial"
	maxAttempts     = 8
	pullBufferBytes = 4 << 20
)

// PullOptions configures Pull.
type PullOptions struct {
	Concurrency int
	// Verify checks every chunk against its SHA-256, both for downloaded data
	// and for data already on disk from an earlier pull.
	Verify       bool
	Log          io.Writer
	ShowProgress bool
	Client       *http.Client
	StallTimeout time.Duration
}

// PullResult summarizes a pull.
type PullResult struct {
	Files           int
	DownloadedBytes int64
	ReusedBytes     int64
	Elapsed         time.Duration
}

type downFile struct {
	manifest.File
	final  string
	target string // final, or final+partialSuffix while incomplete
	check  bool   // target has existing data to verify before downloading

	openOnce  sync.Once
	fh        *os.File
	openErr   error
	remaining atomic.Int32
}

func (d *downFile) open() (*os.File, error) {
	d.openOnce.Do(func() { d.fh, d.openErr = os.OpenFile(d.target, os.O_RDWR, 0) })
	return d.fh, d.openErr
}

type pullJob struct {
	f     *downFile
	chunk int
}

// NewHTTPClient returns a client tuned for many parallel downloads from one
// host: HTTP/1.1 so each request gets its own TCP connection, and a pool
// large enough to keep them all alive.
func NewHTTPClient(concurrency int) *http.Client {
	return &http.Client{Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   30 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		MaxIdleConns:          concurrency * 2,
		MaxIdleConnsPerHost:   concurrency * 2,
		IdleConnTimeout:       90 * time.Second,
		DisableCompression:    true,
		ForceAttemptHTTP2:     false,
		TLSNextProto:          map[string]func(string, *tls.Conn) http.RoundTripper{},
		ReadBufferSize:        256 << 10,
		WriteBufferSize:       64 << 10,
	}}
}

// FetchManifest downloads a share manifest from a URL.
func FetchManifest(ctx context.Context, client *http.Client, url string) (*manifest.Manifest, error) {
	var lastErr error
	for attempt := 1; attempt <= 5; attempt++ {
		m, err := fetchManifestOnce(ctx, client, url)
		if err == nil {
			return m, nil
		}
		var pe *permanentError
		if errors.As(err, &pe) || ctx.Err() != nil {
			return nil, err
		}
		lastErr = err
		if err := sleepCtx(ctx, backoff(attempt)); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func fetchManifestOnce(ctx context.Context, client *http.Client, url string) (*manifest.Manifest, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, &permanentError{err}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch manifest: %w", err)
	}
	defer resp.Body.Close()
	if err := statusError(resp); err != nil {
		return nil, fmt.Errorf("fetch manifest: %w", err)
	}
	m, err := manifest.Decode(resp.Body)
	if err != nil {
		return nil, &permanentError{err}
	}
	return m, nil
}

// Pull downloads every file of m into dest.
//
// Each chunk is a ranged GET streamed straight into its place in a
// preallocated file and hashed on the way, so nothing is buffered beyond a
// few MiB per worker and no second verification pass is needed. Files are
// written as <name>.vastx-partial and renamed when complete. Running Pull
// again after an interruption verifies what's on disk and fetches only the
// chunks that are missing or wrong.
func Pull(ctx context.Context, m *manifest.Manifest, dest string, opts PullOptions) (*PullResult, error) {
	start := time.Now()
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if m.Incomplete {
		return nil, fmt.Errorf("model %q has an unfinished push; run `vastx push` again to complete it", m.Name)
	}
	if m.ExpiresAt != nil && time.Now().After(*m.ExpiresAt) {
		return nil, fmt.Errorf("share link expired at %s; create a new one with `vastx share %s`",
			m.ExpiresAt.Format(time.RFC3339), m.Name)
	}
	for _, f := range m.Files {
		if f.URL == "" {
			return nil, fmt.Errorf("manifest has no download URL for %s", f.Path)
		}
	}
	if opts.Concurrency < 1 {
		opts.Concurrency = DefaultPullConcurrency
	}
	if opts.StallTimeout <= 0 {
		opts.StallTimeout = DefaultStallTimeout
	}
	client := opts.Client
	if client == nil {
		client = NewHTTPClient(opts.Concurrency)
	}
	logw := opts.Log
	if logw == nil {
		logw = io.Discard
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return nil, err
	}

	res := &PullResult{Files: len(m.Files)}
	var plan []*downFile
	var total, need int64
	for _, f := range m.Files {
		final := filepath.Join(dest, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
			return nil, err
		}
		if f.Size == 0 {
			if err := os.WriteFile(final, nil, 0o644); err != nil {
				return nil, err
			}
			continue
		}
		d := &downFile{File: f, final: final, target: final + partialSuffix}
		if fi, err := os.Stat(final); err == nil && fi.Mode().IsRegular() && fi.Size() == f.Size {
			if !opts.Verify {
				res.ReusedBytes += f.Size
				continue
			}
			d.target, d.check = final, true
		} else if fi, err := os.Stat(d.target); err == nil && fi.Mode().IsRegular() && fi.Size() == f.Size {
			d.check = true
		} else {
			need += f.Size
		}
		d.remaining.Store(int32(f.NumChunks()))
		total += f.Size
		plan = append(plan, d)
	}
	if avail, err := freeBytes(dest); err == nil && need > avail {
		return nil, fmt.Errorf("not enough disk space in %s: need %s, %s available",
			dest, FormatBytes(need), FormatBytes(avail))
	}
	for _, d := range plan {
		if d.check {
			continue
		}
		fh, err := os.OpenFile(d.target, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
		if err != nil {
			return nil, err
		}
		err = fh.Truncate(d.Size)
		if cerr := fh.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return nil, err
		}
	}
	defer func() {
		for _, d := range plan {
			if d.fh != nil {
				d.fh.Close()
			}
		}
	}()

	fmt.Fprintf(logw, "pull %s: %d files, %s -> %s\n", m.Name, len(m.Files), FormatBytes(total+res.ReusedBytes), dest)
	prog := NewProgress("pull", total, logw, opts.ShowProgress)
	prog.Start()

	var downloaded, reused atomic.Int64
	g, gctx := errgroup.WithContext(ctx)
	jobs := make(chan pullJob)
	g.Go(func() error {
		defer close(jobs)
		for _, d := range plan {
			for c := range d.NumChunks() {
				select {
				case jobs <- pullJob{d, c}:
				case <-gctx.Done():
					return gctx.Err()
				}
			}
		}
		return nil
	})
	for range opts.Concurrency {
		g.Go(func() error {
			buf := make([]byte, pullBufferBytes)
			for j := range jobs {
				fh, err := j.f.open()
				if err != nil {
					return err
				}
				_, n := j.f.ChunkRange(j.chunk)
				ok := false
				if j.f.check {
					if ok, err = verifyLocal(fh, j.f.File, j.chunk, buf); err != nil {
						return err
					}
				}
				if ok {
					prog.Add(n)
					reused.Add(n)
				} else {
					if err := fetchChunk(gctx, client, j.f, j.chunk, fh, buf, prog, opts); err != nil {
						return err
					}
					downloaded.Add(n)
				}
				if j.f.remaining.Add(-1) == 0 {
					if err := finishDownload(j.f); err != nil {
						return err
					}
				}
			}
			return nil
		})
	}
	err := g.Wait()
	prog.Stop()
	res.DownloadedBytes = downloaded.Load()
	res.ReusedBytes += reused.Load()
	res.Elapsed = time.Since(start)
	if err != nil {
		return res, err
	}
	return res, nil
}

func finishDownload(d *downFile) error {
	if err := d.fh.Close(); err != nil {
		return err
	}
	d.fh = nil
	if d.target != d.final {
		return os.Rename(d.target, d.final)
	}
	return nil
}

func verifyLocal(fh *os.File, f manifest.File, chunk int, buf []byte) (bool, error) {
	off, n := f.ChunkRange(chunk)
	h := sha256.New()
	if _, err := io.CopyBuffer(h, io.NewSectionReader(fh, off, n), buf); err != nil {
		return false, err
	}
	return hex.EncodeToString(h.Sum(nil)) == f.Chunks[chunk], nil
}

func fetchChunk(ctx context.Context, client *http.Client, d *downFile, chunk int, fh *os.File, buf []byte, prog *Progress, opts PullOptions) error {
	for attempt := 1; ; attempt++ {
		err := fetchChunkOnce(ctx, client, d, chunk, fh, buf, prog, opts)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		err = fmt.Errorf("%s chunk %d/%d: %w", d.Path, chunk+1, d.NumChunks(), err)
		var pe *permanentError
		if errors.As(err, &pe) || attempt == maxAttempts {
			return err
		}
		prog.Logf("retrying %v", err)
		if err := sleepCtx(ctx, backoff(attempt)); err != nil {
			return err
		}
	}
}

var errStalled = errors.New("download stalled")

func fetchChunkOnce(ctx context.Context, client *http.Client, d *downFile, chunk int, fh *os.File, buf []byte, prog *Progress, opts PullOptions) (err error) {
	off, n := d.ChunkRange(chunk)
	actx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	req, err := http.NewRequestWithContext(actx, http.MethodGet, d.URL, nil)
	if err != nil {
		return &permanentError{err}
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, off+n-1))
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusPartialContent:
		if cr := resp.Header.Get("Content-Range"); cr != "" && !strings.HasPrefix(cr, fmt.Sprintf("bytes %d-%d/", off, off+n-1)) {
			return fmt.Errorf("server returned range %q, asked for %d-%d", cr, off, off+n-1)
		}
	case resp.StatusCode == http.StatusOK && off == 0 && n == d.Size:
	case resp.StatusCode == http.StatusOK:
		return &permanentError{errors.New("server ignored the range request")}
	default:
		return statusError(resp)
	}

	body := newStallReader(resp.Body, opts.StallTimeout, func() { cancel(errStalled) })
	defer body.stop()
	var h hash.Hash
	if opts.Verify {
		h = sha256.New()
	}
	var written int64
	defer func() {
		if err != nil {
			prog.Add(-written)
		}
	}()
	for written < n {
		k := min(int64(len(buf)), n-written)
		r, rerr := io.ReadFull(body, buf[:k])
		if r > 0 {
			if _, werr := fh.WriteAt(buf[:r], off+written); werr != nil {
				return &permanentError{werr}
			}
			if h != nil {
				h.Write(buf[:r])
			}
			written += int64(r)
			prog.Add(int64(r))
		}
		if rerr != nil {
			if errors.Is(context.Cause(actx), errStalled) {
				return fmt.Errorf("no data for %s", opts.StallTimeout)
			}
			return rerr
		}
	}
	if h != nil {
		if got := hex.EncodeToString(h.Sum(nil)); got != d.Chunks[chunk] {
			return fmt.Errorf("checksum mismatch (got %s, want %s)", got[:12], d.Chunks[chunk][:12])
		}
	}
	return nil
}

// permanentError marks failures that retrying won't fix.
type permanentError struct{ err error }

func (e *permanentError) Error() string { return e.err.Error() }
func (e *permanentError) Unwrap() error { return e.err }

func statusError(resp *http.Response) error {
	if resp.StatusCode < 300 {
		return nil
	}
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	err := fmt.Errorf("HTTP %s: %s", resp.Status, strings.TrimSpace(string(snippet)))
	switch resp.StatusCode {
	case http.StatusForbidden:
		return &permanentError{fmt.Errorf("%w (the link may have expired; create a new one with `vastx share`)", err)}
	case http.StatusNotFound, http.StatusBadRequest, http.StatusUnauthorized, http.StatusRequestedRangeNotSatisfiable:
		return &permanentError{err}
	}
	return err
}

// retryBaseDelay is the first retry delay; it doubles per attempt.
var retryBaseDelay = 500 * time.Millisecond

func backoff(attempt int) time.Duration {
	d := min(retryBaseDelay<<(attempt-1), 30*time.Second)
	return d/2 + rand.N(d/2+1)
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// stallReader calls onStall if no bytes arrive for the timeout.
type stallReader struct {
	r    io.Reader
	last atomic.Int64
	done chan struct{}
	once sync.Once
}

func newStallReader(r io.Reader, timeout time.Duration, onStall func()) *stallReader {
	s := &stallReader{r: r, done: make(chan struct{})}
	s.last.Store(time.Now().UnixNano())
	go func() {
		t := time.NewTicker(max(timeout/4, 10*time.Millisecond))
		defer t.Stop()
		for {
			select {
			case <-s.done:
				return
			case now := <-t.C:
				if now.Sub(time.Unix(0, s.last.Load())) > timeout {
					onStall()
					return
				}
			}
		}
	}()
	return s
}

func (s *stallReader) Read(p []byte) (int, error) {
	n, err := s.r.Read(p)
	if n > 0 {
		s.last.Store(time.Now().UnixNano())
	}
	return n, err
}

func (s *stallReader) stop() { s.once.Do(func() { close(s.done) }) }
