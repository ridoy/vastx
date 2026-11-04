package cli

import (
	"bufio"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"

	"github.com/ridoy/vastx/internal/manifest"
	"github.com/ridoy/vastx/internal/transfer"
)

func newLsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "ls",
		Short: "List models in the bucket",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			st, err := openStore(16)
			if err != nil {
				return err
			}
			names, err := st.ListModels(ctx)
			if err != nil {
				return err
			}
			if len(names) == 0 {
				fmt.Fprintf(cmd.ErrOrStderr(), "no models in bucket %s\n", st.Bucket())
				return nil
			}
			ms := make([]*manifest.Manifest, len(names))
			errs := make([]error, len(names))
			g, gctx := errgroup.WithContext(ctx)
			g.SetLimit(16)
			for i, n := range names {
				g.Go(func() error {
					ms[i], errs[i] = st.GetManifest(gctx, n)
					return nil
				})
			}
			_ = g.Wait()
			if err := ctx.Err(); err != nil {
				return err
			}

			w := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 4, 2, ' ', 0)
			fmt.Fprintln(w, "NAME\tFILES\tSIZE\tPUSHED\tSTATUS")
			for i, n := range names {
				m := ms[i]
				if errs[i] != nil {
					fmt.Fprintf(w, "%s\t-\t-\t-\terror: %v\n", n, errs[i])
					continue
				}
				status := "ok"
				if m.Incomplete {
					status = "push unfinished"
				}
				fmt.Fprintf(w, "%s\t%d\t%s\t%s\t%s\n", n, len(m.Files), transfer.FormatBytes(m.TotalSize()),
					m.CreatedAt.Local().Format("2006-01-02 15:04"), status)
			}
			return w.Flush()
		},
	}
}

func newRmCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "rm <name>",
		Short: "Delete a model from the bucket",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			name := args[0]
			if err := manifest.ValidName(name); err != nil {
				return err
			}
			st, err := openStore(16)
			if err != nil {
				return err
			}
			m, err := st.GetManifest(ctx, name)
			if err != nil {
				return err
			}
			if !yes {
				ok, err := confirm(cmd.InOrStdin(), cmd.ErrOrStderr(),
					fmt.Sprintf("delete %s (%s, %s) from bucket %s?", name, files(len(m.Files)),
						transfer.FormatBytes(m.TotalSize()), st.Bucket()))
				if err != nil {
					return err
				}
				if !ok {
					return fmt.Errorf("aborted")
				}
			}
			start := time.Now()
			n, err := st.DeleteModel(ctx, m)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "deleted %s (%d objects) in %s\n", name, n, time.Since(start).Round(time.Millisecond))
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "don't ask for confirmation")
	return cmd
}

func confirm(in io.Reader, out io.Writer, prompt string) (bool, error) {
	fmt.Fprintf(out, "%s [y/N] ", prompt)
	line, err := bufio.NewReader(in).ReadString('\n')
	if err != nil && line == "" {
		return false, fmt.Errorf("no confirmation given (use --yes to skip the prompt)")
	}
	a := strings.ToLower(strings.TrimSpace(line))
	return a == "y" || a == "yes", nil
}
