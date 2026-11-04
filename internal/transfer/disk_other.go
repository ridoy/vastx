//go:build !(linux || darwin)

package transfer

import "errors"

func freeBytes(string) (int64, error) {
	return 0, errors.New("free space check not supported on this platform")
}
