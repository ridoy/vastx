package cli

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ridoy/vastx/internal/config"
)

func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Show or change storage settings",
		Long: `Show or change storage settings.

Settings live in a JSON file (see 'vastx config path'). Each one can also be
given as an environment variable, which takes precedence over the file.

For Cloudflare R2, set account_id, access_key_id, secret_access_key and
bucket. Create the keys under R2 > Manage API tokens with "Object Read &
Write" on the bucket. For other S3-compatible stores, set endpoint (and
region if it isn't "auto") instead of account_id.`,
	}
	cmd.AddCommand(&cobra.Command{
		Use:   "show",
		Short: "Print the effective settings (secrets redacted)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			p, err := config.Path()
			if err != nil {
				return err
			}
			c, err := config.Load()
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "file: %s\n\n", p)
			w := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
			fmt.Fprintln(w, "KEY\tVALUE\tENV")
			for _, e := range c.Entries() {
				v := e.Value
				if v == "" {
					v = "-"
				}
				if os.Getenv(e.Env) != "" {
					v += " (from env)"
				}
				fmt.Fprintf(w, "%s\t%s\t%s\n", e.Key, v, e.Env)
			}
			if err := w.Flush(); err != nil {
				return err
			}
			if err := c.Validate(); err != nil {
				fmt.Fprintf(out, "\n%v\n", err)
			} else {
				fmt.Fprintf(out, "\nendpoint: %s\n", c.EndpointURL())
			}
			return nil
		},
	})
	cmd.AddCommand(&cobra.Command{
		Use:       "set <key> <value>",
		Short:     "Set a value in the config file",
		Args:      cobra.ExactArgs(2),
		ValidArgs: config.Keys(),
		RunE: func(cmd *cobra.Command, args []string) error {
			p, err := config.Path()
			if err != nil {
				return err
			}
			c, err := config.LoadFile(p)
			if err != nil {
				return err
			}
			if err := c.Set(args[0], args[1]); err != nil {
				return err
			}
			if err := config.Save(p, c); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "set %s in %s\n", args[0], p)
			return nil
		},
	})
	cmd.AddCommand(&cobra.Command{
		Use:   "path",
		Short: "Print the config file location",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			p, err := config.Path()
			if err != nil {
				return err
			}
			fmt.Fprintln(cmd.OutOrStdout(), p)
			return nil
		},
	})
	return cmd
}
