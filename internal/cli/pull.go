package cli

import (
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ridoy/vastx/internal/manifest"
	"github.com/ridoy/vastx/internal/transfer"
)

func newPullCmd() *cobra.Command {
	var (
		concurrency int
		noVerify    bool
	)
	cmd := &cobra.Command{
		Use:   "pull <name | share-url> [dest]",
		Short: "Download a model with parallel ranged requests",
		Long: `Download a model into dest (default: ./<last segment of the name>).

The source is either a share URL from 'vastx share', which needs no
credentials, or a model name, which needs storage credentials configured.

Every chunk is verified against the SHA-256 recorded at push time. Files are
written as <file>.vastx-partial and renamed once complete. If a pull is
interrupted, run the same command again: data already on disk is verified
and only missing or bad chunks are downloaded.`,
		Example: `  vastx pull 'https://<account>.r2.cloudflarestorage.com/...' /workspace/models/llama
  vastx pull meta-llama/Llama-3.1-8B`,
		Args: cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			src := args[0]
			client := transfer.NewHTTPClient(concurrency)
			var m *manifest.Manifest
			var err error
			if strings.HasPrefix(src, "https://") || strings.HasPrefix(src, "http://") {
				m, err = transfer.FetchManifest(ctx, client, src)
			} else {
				m, err = presignedManifest(cmd, src)
			}
			if err != nil {
				return err
			}
			dest := path.Base(m.Name)
			if len(args) == 2 {
				dest = args[1]
			}
			res, err := transfer.Pull(ctx, m, dest, transfer.PullOptions{
				Concurrency:  concurrency,
				Verify:       !noVerify,
				Log:          cmd.ErrOrStderr(),
				ShowProgress: !quiet(cmd),
				Client:       client,
			})
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "pulled %s into %s: %s, downloaded %s",
				m.Name, dest, files(res.Files), transfer.FormatBytes(res.DownloadedBytes))
			if res.ReusedBytes > 0 {
				fmt.Fprintf(cmd.OutOrStdout(), ", %s already present", transfer.FormatBytes(res.ReusedBytes))
			}
			fmt.Fprintf(cmd.OutOrStdout(), " in %s\n", res.Elapsed.Round(time.Second/10))
			return nil
		},
	}
	f := cmd.Flags()
	f.IntVarP(&concurrency, "concurrency", "j", transfer.DefaultPullConcurrency, "parallel ranged requests")
	f.BoolVar(&noVerify, "no-verify", false, "skip SHA-256 checks and trust files already at the right size")
	return cmd
}

// presignedManifest fetches a model's manifest with credentials and
// presigns its files locally, so pull by name and pull by share URL use the
// same download path.
func presignedManifest(cmd *cobra.Command, name string) (*manifest.Manifest, error) {
	if err := manifest.ValidName(name); err != nil {
		return nil, err
	}
	st, err := openStore(16)
	if err != nil {
		return nil, err
	}
	m, err := st.GetManifest(cmd.Context(), name)
	if err != nil {
		return nil, err
	}
	return st.Presign(cmd.Context(), m, 24*time.Hour)
}
