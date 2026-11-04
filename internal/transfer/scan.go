package transfer

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"time"
)

// DefaultExcludes are skipped on push unless overridden.
var DefaultExcludes = []string{".git", ".DS_Store", ".cache/huggingface", "*" + partialSuffix}

// LocalFile is a file found by Scan.
type LocalFile struct {
	Abs     string
	Rel     string // slash-separated, relative to the scan root
	Size    int64
	ModTime time.Time
}

// Scan lists the regular files under root, or root itself if it's a file.
// Symlinks are followed, since Hugging Face cache snapshots are symlinks
// into a blob store. Excludes are path.Match patterns tested against both
// the relative path and the base name; a matching directory is skipped whole.
func Scan(root string, excludes []string) ([]LocalFile, error) {
	fi, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !fi.IsDir() {
		if !fi.Mode().IsRegular() {
			return nil, fmt.Errorf("%s is not a regular file or directory", root)
		}
		abs, err := filepath.Abs(root)
		if err != nil {
			return nil, err
		}
		return []LocalFile{{Abs: abs, Rel: filepath.Base(abs), Size: fi.Size(), ModTime: fi.ModTime()}}, nil
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	var out []LocalFile
	if err := walk(abs, "", map[string]bool{}, excludes, &out); err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Rel < out[j].Rel })
	return out, nil
}

func walk(dir, rel string, visited map[string]bool, excludes []string, out *[]LocalFile) error {
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return err
	}
	if visited[real] {
		return fmt.Errorf("symlink cycle at %s", dir)
	}
	visited[real] = true
	defer delete(visited, real)

	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		r := path.Join(rel, e.Name())
		if excluded(r, e.Name(), excludes) {
			continue
		}
		p := filepath.Join(dir, e.Name())
		fi, err := os.Stat(p) // follows symlinks
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		switch {
		case fi.IsDir():
			if err := walk(p, r, visited, excludes, out); err != nil {
				return err
			}
		case fi.Mode().IsRegular():
			*out = append(*out, LocalFile{Abs: p, Rel: r, Size: fi.Size(), ModTime: fi.ModTime()})
		}
	}
	return nil
}

func excluded(rel, base string, patterns []string) bool {
	for _, p := range patterns {
		if ok, _ := path.Match(p, rel); ok {
			return true
		}
		if ok, _ := path.Match(p, base); ok {
			return true
		}
	}
	return false
}
