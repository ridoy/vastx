package store

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ridoy/vastx/internal/manifest"
)

const manifestSuffix = "/" + manifest.MetaDir + "/manifest.json"

// GetManifest fetches a model's manifest. It returns an error wrapping
// ErrNotFound if the model doesn't exist.
func (s *Store) GetManifest(ctx context.Context, name string) (*manifest.Manifest, error) {
	b, err := s.GetBytes(ctx, ManifestKey(name))
	if err != nil {
		if IsNotFound(err) {
			return nil, fmt.Errorf("model %q: %w", name, ErrNotFound)
		}
		return nil, err
	}
	m, err := manifest.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", name, err)
	}
	return m, nil
}

// PutManifest writes a model's manifest.
func (s *Store) PutManifest(ctx context.Context, m *manifest.Manifest) error {
	b, err := m.Encode()
	if err != nil {
		return err
	}
	return s.PutBytes(ctx, ManifestKey(m.Name), b, "application/json")
}

// ListModels returns the names of all models in the bucket.
func (s *Store) ListModels(ctx context.Context) ([]string, error) {
	var names []string
	err := s.List(ctx, "", func(key string, _ int64) error {
		if strings.HasSuffix(key, manifestSuffix) && !strings.HasPrefix(key, manifest.MetaDir+"/") {
			names = append(names, strings.TrimSuffix(key, manifestSuffix))
		}
		return nil
	})
	sort.Strings(names)
	return names, err
}

// CheckNameFree makes sure a model name doesn't nest inside another model or
// contain one, since their objects would share a key prefix.
func (s *Store) CheckNameFree(ctx context.Context, name string) error {
	segs := strings.Split(name, "/")
	for i := 1; i < len(segs); i++ {
		anc := strings.Join(segs[:i], "/")
		_, ok, err := s.Head(ctx, ManifestKey(anc))
		if err != nil {
			return err
		}
		if ok {
			return fmt.Errorf("model %q already exists, so %q can't be nested inside it", anc, name)
		}
	}
	own := ManifestKey(name)
	return s.List(ctx, name+"/", func(key string, _ int64) error {
		if strings.HasSuffix(key, manifestSuffix) && key != own {
			return fmt.Errorf("model %q exists inside %q; pick a different name",
				strings.TrimSuffix(key, manifestSuffix), name)
		}
		return nil
	})
}

// DeleteModel removes a model's files and metadata. It only deletes keys the
// manifest lists, plus the model's metadata directory.
func (s *Store) DeleteModel(ctx context.Context, m *manifest.Manifest) (int, error) {
	keys := make([]string, 0, len(m.Files)+1)
	for _, f := range m.Files {
		keys = append(keys, ObjectKey(m.Name, f.Path))
	}
	err := s.List(ctx, m.Name+"/"+manifest.MetaDir+"/", func(key string, _ int64) error {
		if key != ManifestKey(m.Name) {
			keys = append(keys, key)
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	if err := s.DeleteKeys(ctx, keys); err != nil {
		return 0, err
	}
	// Remove the manifest last so a failed delete can be retried.
	return len(keys) + 1, s.Delete(ctx, ManifestKey(m.Name))
}

// DeleteKeys deletes objects concurrently.
func (s *Store) DeleteKeys(ctx context.Context, keys []string) error {
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(16)
	for _, k := range keys {
		g.Go(func() error { return s.Delete(gctx, k) })
	}
	return g.Wait()
}

// Presign returns a copy of m with a presigned GET URL on every file.
func (s *Store) Presign(ctx context.Context, m *manifest.Manifest, ttl time.Duration) (*manifest.Manifest, error) {
	if ttl <= 0 || ttl > MaxPresignTTL {
		return nil, fmt.Errorf("link lifetime must be between 1s and %s", MaxPresignTTL)
	}
	if m.Incomplete {
		return nil, fmt.Errorf("model %q has an unfinished push; run `vastx push` again to complete it", m.Name)
	}
	expires := time.Now().Add(ttl).UTC().Truncate(time.Second)
	out := *m
	out.ExpiresAt = &expires
	out.Files = make([]manifest.File, len(m.Files))
	for i, f := range m.Files {
		u, err := s.PresignGet(ctx, ObjectKey(m.Name, f.Path), ttl)
		if err != nil {
			return nil, err
		}
		f.URL = u
		out.Files[i] = f
	}
	return &out, nil
}

// Share is a published share manifest.
type Share struct {
	URL       string
	ExpiresAt time.Time
}

// CreateShare uploads a presigned copy of m and returns a presigned URL to it.
// Anyone holding the URL can download the model until it expires, without
// any storage credentials.
func (s *Store) CreateShare(ctx context.Context, m *manifest.Manifest, ttl time.Duration) (*Share, error) {
	shared, err := s.Presign(ctx, m, ttl)
	if err != nil {
		return nil, err
	}
	b, err := shared.Encode()
	if err != nil {
		return nil, err
	}
	var id [12]byte
	if _, err := rand.Read(id[:]); err != nil {
		return nil, err
	}
	key := fmt.Sprintf("%s/shares/%s-%s.json", manifest.MetaDir,
		strings.ReplaceAll(m.Name, "/", "--"), hex.EncodeToString(id[:]))
	if err := s.PutBytes(ctx, key, b, "application/json"); err != nil {
		return nil, err
	}
	u, err := s.PresignGet(ctx, key, ttl)
	if err != nil {
		return nil, err
	}
	return &Share{URL: u, ExpiresAt: *shared.ExpiresAt}, nil
}

// UploadBinary stores a vastx binary under a content-addressed key, skipping
// the upload if it's already there, and returns a presigned URL to it.
func (s *Store) UploadBinary(ctx context.Context, localPath, objName string, ttl time.Duration) (string, error) {
	b, err := os.ReadFile(localPath)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	key := fmt.Sprintf("%s/bin/%x/%s", manifest.MetaDir, sum[:8], objName)
	size, ok, err := s.Head(ctx, key)
	if err != nil {
		return "", err
	}
	if !ok || size != int64(len(b)) {
		if err := s.PutBytes(ctx, key, b, "application/octet-stream"); err != nil {
			return "", err
		}
	}
	return s.PresignGet(ctx, key, ttl)
}
