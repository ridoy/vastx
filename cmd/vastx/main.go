// Command vastx is a CLI for renting cloud GPUs for LLM work.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ridoy/vastx/internal/cli"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	err := cli.NewRootCmd().ExecuteContext(ctx)
	if err == nil {
		return
	}
	if errors.Is(err, context.Canceled) && ctx.Err() != nil {
		fmt.Fprintln(os.Stderr, "interrupted; run the same command again to resume")
		os.Exit(130)
	}
	fmt.Fprintf(os.Stderr, "vastx: %v\n", err)
	os.Exit(1)
}
