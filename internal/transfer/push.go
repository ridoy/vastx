package transfer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ridoy/vastx/internal/manifest"
	"github.com/ridoy/vastx/internal/store"
)

const (
	// DefaultChunkSize is the part size for uploads and the range size for
	// downloads.
	DefaultChunkSize = 64 << 20
	// DefaultPushConcurrency is the number of parts uploaded at once. Push
	// holds one chunk in memory per worker.
	DefaultPushConcurrency = 8

	checkpointInterval = 5 * time.Second
)

// PushOptions configures Push.
type PushOptions struct {
	Concurrency  int
	ChunkSize    int64
	Force        bool // re-upload files that look unchanged
	Excludes     []string
	Log          io.Writer
	ShowProgress bool
}

// PushResult summarizes a push.
type PushResult struct {
	Manifest       *manifest.Manifest
	UploadedFiles  int
	SkippedFiles   int
	UploadedBytes  int64
	DeletedObjects int
	Elapsed        time.Duration
}

type upFile struct {
	idx       int
	lf        LocalFile
	key       string
	chunkSize int64
	jobs      int
	uploadID  string
	hashes    []string
	etags     []string
	remaining atomic.Int32
}

func (f *upFile) entry() manifest.File {
	return manifest.File{Path: f.lf.Rel, Size: f.lf.Size, ModTime: f.lf.ModTime, ChunkSize: f.chunkSize, Chunks: f.hashes}
}

type pushJob struct {
	f     *upFile
	chunk int
}

// Push uploads src (a file or directory) as model name.
//
// Every file is split into chunks that are hashed and uploaded as multipart
// parts by a pool of workers, so large files and many small files both keep
// all connections busy. Files whose size and mtime match the previous push
// are skipped. The manifest is checkpointed as files finish, so an
// interrupted push resumes at file granularity.
func Push(ctx context.Context, st *store.Store, src, name string, opts PushOptions) (*PushResult, error) {
	start := time.Now()
	if err := manifest.ValidName(name); err != nil {
		return nil, err
	}
	if opts.Concurrency < 1 {
		opts.Concurrency = DefaultPushConcurrency
	}
	if opts.ChunkSize == 0 {
		opts.ChunkSize = DefaultChunkSize
	}
	if opts.ChunkSize < manifest.MinChunkSize {
		return nil, fmt.Errorf("chunk size must be at least %s", FormatBytes(manifest.MinChunkSize))
	}
	logw := opts.Log
	if logw == nil {
		logw = io.Discard
	}

	files, err := Scan(src, opts.Excludes)
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no files to upload in %s", src)
	}
	if err := st.CheckNameFree(ctx, name); err != nil {
		return nil, err
	}
	old, err := st.GetManifest(ctx, name)
	if err != nil && !store.IsNotFound(err) {
		return nil, err
	}

	ck := &checkpointer{st: st, name: name, entries: make([]manifest.File, len(files)), done: make([]bool, len(files))}
	if old != nil && !opts.Force {
		if err := ck.markUnchanged(ctx, files, old, opts.ChunkSize); err != nil {
			return nil, err
		}
	}

	var todo []*upFile
	var total, maxChunk int64
	for i, lf := range files {
		if ck.done[i] {
			continue
		}
		cs := manifest.ChunkSizeFor(lf.Size, opts.ChunkSize)
		uf := &upFile{idx: i, lf: lf, key: store.ObjectKey(name, lf.Rel), chunkSize: cs}
		n := uf.entry().NumChunks()
		uf.hashes = make([]string, n)
		uf.etags = make([]string, n)
		uf.jobs = max(n, 1) // an empty file still needs one PutObject
		uf.remaining.Store(int32(uf.jobs))
		todo = append(todo, uf)
		total += lf.Size
		maxChunk = max(maxChunk, min(cs, lf.Size))
	}
	res := &PushResult{UploadedFiles: len(todo), SkippedFiles: len(files) - len(todo), UploadedBytes: total}
	fmt.Fprintf(logw, "push %s: %d files, %s to upload", name, len(files), FormatBytes(total))
	if res.SkippedFiles > 0 {
		fmt.Fprintf(logw, " (%d unchanged files skipped)", res.SkippedFiles)
	}
	fmt.Fprintln(logw)

	prog := NewProgress("push", total, logw, opts.ShowProgress)
	prog.Start()

	var openMu sync.Mutex
	open := map[*upFile]bool{}
	g, gctx := errgroup.WithContext(ctx)
	jobs := make(chan pushJob)
	g.Go(func() error {
		defer close(jobs)
		for _, uf := range todo {
			if uf.jobs > 1 {
				id, err := st.CreateMultipart(gctx, uf.key)
				if err != nil {
					return err
				}
				uf.uploadID = id
				openMu.Lock()
				open[uf] = true
				openMu.Unlock()
			}
			for c := 0; c < uf.jobs; c++ {
				select {
				case jobs <- pushJob{uf, c}:
				case <-gctx.Done():
					return gctx.Err()
				}
			}
		}
		return nil
	})
	for range opts.Concurrency {
		g.Go(func() error {
			buf := make([]byte, maxChunk)
			for j := range jobs {
				n, err := pushChunk(gctx, st, j, buf)
				if err != nil {
					return err
				}
				prog.Add(n)
				if j.f.remaining.Add(-1) != 0 {
					continue
				}
				if err := finishUpload(gctx, st, j.f); err != nil {
					return err
				}
				openMu.Lock()
				delete(open, j.f)
				openMu.Unlock()
				ck.complete(gctx, j.f.idx, j.f.entry())
			}
			return nil
		})
	}
	err = g.Wait()
	prog.Stop()
	if err != nil {
		cctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		defer cancel()
		for uf := range open {
			_ = st.AbortMultipart(cctx, uf.key, uf.uploadID)
		}
		// Record finished files so the next push skips them.
		_ = ck.write(cctx, true)
		return nil, err
	}

	m := ck.manifest(false)
	if err := st.PutManifest(ctx, m); err != nil {
		return nil, err
	}
	if old != nil {
		keep := make(map[string]bool, len(m.Files))
		for _, f := range m.Files {
			keep[f.Path] = true
		}
		var stale []string
		for _, f := range old.Files {
			if !keep[f.Path] {
				stale = append(stale, store.ObjectKey(name, f.Path))
			}
		}
		if err := st.DeleteKeys(ctx, stale); err != nil {
			return nil, fmt.Errorf("remove files deleted since the last push: %w", err)
		}
		res.DeletedObjects = len(stale)
	}
	res.Manifest = m
	res.Elapsed = time.Since(start)
	return res, nil
}

// pushChunk uploads one chunk and returns its length.
func pushChunk(ctx context.Context, st *store.Store, j pushJob, buf []byte) (int64, error) {
	f := j.f
	var off, n int64
	if f.lf.Size > 0 {
		off, n = f.entry().ChunkRange(j.chunk)
	}
	b := buf[:n]
	if n > 0 {
		fh, err := os.Open(f.lf.Abs)
		if err != nil {
			return 0, err
		}
		r, err := fh.ReadAt(b, off)
		fh.Close()
		if int64(r) != n {
			if err == nil || errors.Is(err, io.EOF) {
				err = fmt.Errorf("%s shrank during upload", f.lf.Abs)
			}
			return 0, err
		}
		sum := sha256.Sum256(b)
		f.hashes[j.chunk] = hex.EncodeToString(sum[:])
	}
	if f.uploadID == "" {
		return n, st.PutBytes(ctx, f.key, b, "")
	}
	etag, err := st.UploadPart(ctx, f.key, f.uploadID, int32(j.chunk+1), b)
	if err != nil {
		return 0, err
	}
	f.etags[j.chunk] = etag
	return n, nil
}

func finishUpload(ctx context.Context, st *store.Store, f *upFile) error {
	if f.uploadID != "" {
		if err := st.CompleteMultipart(ctx, f.key, f.uploadID, f.etags); err != nil {
			return err
		}
	}
	fi, err := os.Stat(f.lf.Abs)
	if err != nil {
		return err
	}
	if fi.Size() != f.lf.Size || !fi.ModTime().Equal(f.lf.ModTime) {
		return fmt.Errorf("%s changed during upload; push again once it's stable", f.lf.Abs)
	}
	return nil
}

// checkpointer tracks finished files and periodically writes an incomplete
// manifest listing them.
type checkpointer struct {
	st      *store.Store
	name    string
	mu      sync.Mutex
	entries []manifest.File
	done    []bool
	last    time.Time
}

// markUnchanged reuses manifest entries for files whose size, mtime and
// chunk size match the previous push and whose object is still present.
func (c *checkpointer) markUnchanged(ctx context.Context, files []LocalFile, old *manifest.Manifest, chunkSize int64) error {
	prev := make(map[string]manifest.File, len(old.Files))
	for _, f := range old.Files {
		prev[f.Path] = f
	}
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(16)
	for i, lf := range files {
		pf, ok := prev[lf.Rel]
		if !ok || pf.Size != lf.Size || !pf.ModTime.Equal(lf.ModTime) || pf.ChunkSize != manifest.ChunkSizeFor(lf.Size, chunkSize) {
			continue
		}
		g.Go(func() error {
			size, exists, err := c.st.Head(gctx, store.ObjectKey(c.name, lf.Rel))
			if err != nil {
				return err
			}
			if exists && size == lf.Size {
				pf.URL = ""
				c.entries[i], c.done[i] = pf, true
			}
			return nil
		})
	}
	return g.Wait()
}

func (c *checkpointer) complete(ctx context.Context, i int, e manifest.File) {
	c.mu.Lock()
	c.entries[i], c.done[i] = e, true
	due := time.Since(c.last) >= checkpointInterval
	if due {
		c.last = time.Now()
	}
	c.mu.Unlock()
	if due {
		// Best effort: a failed checkpoint only costs re-uploading on resume.
		_ = c.write(ctx, true)
	}
}

func (c *checkpointer) write(ctx context.Context, incomplete bool) error {
	return c.st.PutManifest(ctx, c.manifest(incomplete))
}

func (c *checkpointer) manifest(incomplete bool) *manifest.Manifest {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := &manifest.Manifest{
		Version:    manifest.FormatVersion,
		Name:       c.name,
		CreatedAt:  time.Now().UTC().Truncate(time.Second),
		Incomplete: incomplete,
		Files:      []manifest.File{},
	}
	for i, e := range c.entries {
		if c.done[i] {
			m.Files = append(m.Files, e)
		}
	}
	sort.Slice(m.Files, func(i, j int) bool { return m.Files[i].Path < m.Files[j].Path })
	return m
}
