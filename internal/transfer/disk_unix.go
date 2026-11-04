//go:build linux || darwin

package transfer

import "syscall"

// freeBytes returns the space available to unprivileged users on dir's
// filesystem.
func freeBytes(dir string) (int64, error) {
	var s syscall.Statfs_t
	if err := syscall.Statfs(dir, &s); err != nil {
		return 0, err
	}
	return int64(s.Bavail) * int64(s.Bsize), nil
}
