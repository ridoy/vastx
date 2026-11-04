package transfer

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Progress prints transfer progress. On a terminal it redraws one line; in
// logs (e.g. a vast.ai onstart script) it prints a line every 10 seconds.
type Progress struct {
	label   string
	total   int64
	done    atomic.Int64
	out     io.Writer
	enabled bool
	tty     bool

	mu      sync.Mutex
	start   time.Time
	samples []sample
	stopc   chan struct{}
	wg      sync.WaitGroup
}

type sample struct {
	t    time.Time
	done int64
}

// NewProgress creates a progress reporter. Nothing is printed unless enabled.
func NewProgress(label string, total int64, out io.Writer, enabled bool) *Progress {
	return &Progress{label: label, total: total, out: out, enabled: enabled && out != nil, tty: isTerminal(out)}
}

// Add records n transferred bytes. n may be negative to undo a failed attempt.
func (p *Progress) Add(n int64) { p.done.Add(n) }

// Start begins periodic printing.
func (p *Progress) Start() {
	p.start = time.Now()
	if !p.enabled {
		return
	}
	interval := 10 * time.Second
	if p.tty {
		interval = 250 * time.Millisecond
	}
	p.stopc = make(chan struct{})
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-p.stopc:
				return
			case <-t.C:
				p.print()
			}
		}
	}()
}

// Stop ends periodic printing and prints a final summary line.
func (p *Progress) Stop() {
	if !p.enabled || p.stopc == nil {
		return
	}
	close(p.stopc)
	p.wg.Wait()
	p.stopc = nil
	elapsed := time.Since(p.start)
	done := p.done.Load()
	line := fmt.Sprintf("%s  %s in %s (%s/s)", p.label, FormatBytes(done), formatDuration(elapsed),
		FormatBytes(int64(float64(done)/max(elapsed.Seconds(), 0.001))))
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.tty {
		fmt.Fprintf(p.out, "\r\033[K%s\n", line)
	} else {
		fmt.Fprintln(p.out, line)
	}
}

// Logf prints a message without garbling the progress line.
func (p *Progress) Logf(format string, args ...any) {
	if p.out == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.enabled && p.tty {
		fmt.Fprint(p.out, "\r\033[K")
	}
	fmt.Fprintf(p.out, format+"\n", args...)
}

func (p *Progress) print() {
	now := time.Now()
	done := p.done.Load()
	p.mu.Lock()
	defer p.mu.Unlock()

	window := 5 * time.Second
	if !p.tty {
		window = 30 * time.Second
	}
	p.samples = append(p.samples, sample{now, done})
	for len(p.samples) > 2 && now.Sub(p.samples[1].t) > window {
		p.samples = p.samples[1:]
	}
	var rate float64
	if first := p.samples[0]; now.Sub(first.t) > 0 && len(p.samples) > 1 {
		rate = float64(done-first.done) / now.Sub(first.t).Seconds()
	} else if el := now.Sub(p.start).Seconds(); el > 0 {
		rate = float64(done) / el
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s  %s / %s", p.label, FormatBytes(done), FormatBytes(p.total))
	if p.total > 0 {
		fmt.Fprintf(&b, "  %3.0f%%", 100*float64(done)/float64(p.total))
	}
	fmt.Fprintf(&b, "  %s/s", FormatBytes(int64(rate)))
	if rate > 0 && done < p.total {
		eta := time.Duration(float64(p.total-done) / rate * float64(time.Second))
		fmt.Fprintf(&b, "  ETA %s", formatDuration(eta))
	}
	if p.tty {
		fmt.Fprintf(p.out, "\r\033[K%s", b.String())
	} else {
		fmt.Fprintln(p.out, b.String())
	}
}

func isTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

func formatDuration(d time.Duration) string {
	if d < time.Second {
		return d.Round(time.Millisecond).String()
	}
	return d.Round(time.Second).String()
}
