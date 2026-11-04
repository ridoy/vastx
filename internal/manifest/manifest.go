// Package manifest defines the description of a model stored in the bucket.
//
// A manifest lists every file of a model with its size and a SHA-256 per
// fixed-size chunk. Chunks double as multipart upload parts on push and as
// ranged GET requests on pull, so each chunk can be verified and retried on
// its own.
package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path"
	"regexp"
	"strings"
	"time"
)

const (
	// FormatVersion is the manifest schema version written by this build.
	FormatVersion = 1
	// MinChunkSize is the smallest multipart part size S3 and R2 accept.
	MinChunkSize = 5 << 20
	// MaxParts is the S3/R2 limit on parts per multipart upload.
	MaxParts = 10000
	// MetaDir is the reserved directory holding vastx metadata.
	MetaDir = ".vastx"
)

// Manifest describes one model: a named set of files.
type Manifest struct {
	Version   int       `json:"version"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	// Incomplete is set on checkpoints written while a push is running.
	Incomplete bool `json:"incomplete,omitempty"`
	// ExpiresAt is set on share manifests, whose file URLs are presigned.
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
	Files     []File     `json:"files"`
}

// File is one file of a model.
type File struct {
	// Path is slash-separated and relative to the model root.
	Path      string    `json:"path"`
	Size      int64     `json:"size"`
	ModTime   time.Time `json:"mtime"`
	ChunkSize int64     `json:"chunk_size"`
	// Chunks holds the hex SHA-256 of each chunk, in order.
	Chunks []string `json:"chunks"`
	// URL is a presigned GET URL. Only present in share manifests.
	URL string `json:"url,omitempty"`
}

// NumChunks returns how many chunks a file of this size and chunk size has.
func (f File) NumChunks() int {
	if f.Size == 0 || f.ChunkSize <= 0 {
		return 0
	}
	return int((f.Size + f.ChunkSize - 1) / f.ChunkSize)
}

// ChunkRange returns the byte offset and length of chunk i.
func (f File) ChunkRange(i int) (off, n int64) {
	off = int64(i) * f.ChunkSize
	n = f.ChunkSize
	if off+n > f.Size {
		n = f.Size - off
	}
	return off, n
}

// ChunkSizeFor returns the chunk size to use for a file of the given size:
// the preferred size, grown in whole MiB if needed to stay within MaxParts.
func ChunkSizeFor(size, preferred int64) int64 {
	cs := preferred
	if minSize := (size + MaxParts - 1) / MaxParts; minSize > cs {
		const mib = 1 << 20
		cs = (minSize + mib - 1) / mib * mib
	}
	return cs
}

// TotalSize returns the sum of all file sizes.
func (m *Manifest) TotalSize() int64 {
	var t int64
	for _, f := range m.Files {
		t += f.Size
	}
	return t
}

// Validate checks structural invariants. Manifests are read from the network,
// so this also guards against paths escaping the destination directory.
func (m *Manifest) Validate() error {
	if m.Version != FormatVersion {
		return fmt.Errorf("unsupported manifest version %d (this build reads version %d)", m.Version, FormatVersion)
	}
	if err := ValidName(m.Name); err != nil {
		return err
	}
	seen := make(map[string]bool, len(m.Files))
	for _, f := range m.Files {
		if err := ValidPath(f.Path); err != nil {
			return err
		}
		if seen[f.Path] {
			return fmt.Errorf("manifest lists %q twice", f.Path)
		}
		seen[f.Path] = true
		if f.Size < 0 {
			return fmt.Errorf("%s: negative size", f.Path)
		}
		if f.Size > 0 && f.ChunkSize <= 0 {
			return fmt.Errorf("%s: invalid chunk size %d", f.Path, f.ChunkSize)
		}
		if len(f.Chunks) != f.NumChunks() {
			return fmt.Errorf("%s: has %d chunk hashes, want %d", f.Path, len(f.Chunks), f.NumChunks())
		}
		for _, c := range f.Chunks {
			if len(c) != 64 {
				return fmt.Errorf("%s: malformed chunk hash %q", f.Path, c)
			}
		}
	}
	// A file "a" and a file "a/b" can't both exist on disk.
	for p := range seen {
		for dir := path.Dir(p); dir != "."; dir = path.Dir(dir) {
			if seen[dir] {
				return fmt.Errorf("manifest has both file %q and file %q inside it", dir, p)
			}
		}
	}
	return nil
}

var segmentRE = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// ValidName checks a model name. Names are bucket key prefixes and may
// contain slashes, e.g. "meta-llama/Llama-3.1-8B".
func ValidName(name string) error {
	if name == "" {
		return errors.New("model name is empty")
	}
	for _, seg := range strings.Split(name, "/") {
		if seg == "" || seg == "." || seg == ".." || seg == MetaDir || !segmentRE.MatchString(seg) {
			return fmt.Errorf("invalid model name %q: use letters, digits, '.', '_', '-' and '/' between segments", name)
		}
	}
	return nil
}

// ValidPath checks a file path inside a model.
func ValidPath(p string) error {
	bad := p == "" ||
		strings.HasPrefix(p, "/") ||
		strings.Contains(p, "\\") ||
		strings.ContainsRune(p, 0) ||
		path.Clean(p) != p
	if !bad {
		segs := strings.Split(p, "/")
		if segs[0] == MetaDir {
			bad = true
		}
		for _, seg := range segs {
			if seg == ".." || seg == "." {
				bad = true
			}
		}
	}
	if bad {
		return fmt.Errorf("invalid file path %q in manifest", p)
	}
	return nil
}

// Decode reads and validates a manifest.
func Decode(r io.Reader) (*Manifest, error) {
	var m Manifest
	if err := json.NewDecoder(r).Decode(&m); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

// Encode serializes a manifest.
func (m *Manifest) Encode() ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}
