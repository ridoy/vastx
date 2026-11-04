// Package cli implements the vastx command line.
package cli

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/ridoy/vastx/internal/config"
	"github.com/ridoy/vastx/internal/store"
)

// Version is set at build time.
var Version = "dev"

// NewRootCmd builds the command tree.
func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "vastx",
		Short: "Tooling for renting cloud GPUs for LLM training and inference",
		Long: `vastx moves model weights between your machine, a Cloudflare R2 bucket and
rented GPU instances.

Push weights to R2 once from your machine, then pull them onto any instance
with many parallel ranged downloads. A share link lets an instance download
a model without holding any storage credentials.`,
		Version:       Version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.PersistentFlags().BoolP("quiet", "q", false, "don't print progress")
	root.AddCommand(newPushCmd(), newPullCmd(), newShareCmd(), newLsCmd(), newRmCmd(), newConfigCmd())
	return root
}

func openStore(conns int) (*store.Store, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	return store.New(cfg, conns)
}

func files(n int) string {
	if n == 1 {
		return "1 file"
	}
	return fmt.Sprintf("%d files", n)
}

func quiet(cmd *cobra.Command) bool {
	q, _ := cmd.Flags().GetBool("quiet")
	return q
}
