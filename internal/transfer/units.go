package transfer

import (
	"fmt"
	"strconv"
	"strings"
)

// FormatBytes renders a byte count with binary units.
func FormatBytes(n int64) string {
	const unit = 1024
	if n < unit && n > -unit {
		return fmt.Sprintf("%d B", n)
	}
	v := float64(n)
	suffixes := []string{"KiB", "MiB", "GiB", "TiB", "PiB"}
	i := -1
	for (v >= unit || v <= -unit) && i < len(suffixes)-1 {
		v /= unit
		i++
	}
	return fmt.Sprintf("%.1f %s", v, suffixes[i])
}

// ParseSize parses sizes like "64MiB", "64M", "1g" or "1048576". All units
// are binary (K = 1024).
func ParseSize(s string) (int64, error) {
	t := strings.ToLower(strings.TrimSpace(s))
	i := 0
	for i < len(t) && (t[i] >= '0' && t[i] <= '9') {
		i++
	}
	num, unit := t[:i], strings.TrimSpace(t[i:])
	n, err := strconv.ParseInt(num, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	var mult int64
	switch strings.TrimSuffix(strings.TrimSuffix(unit, "b"), "i") {
	case "":
		mult = 1
	case "k":
		mult = 1 << 10
	case "m":
		mult = 1 << 20
	case "g":
		mult = 1 << 30
	default:
		return 0, fmt.Errorf("invalid size %q: unknown unit %q", s, unit)
	}
	return n * mult, nil
}
