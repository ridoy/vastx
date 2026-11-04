package cli

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ridoy/vastx/internal/manifest"
	"github.com/ridoy/vastx/internal/transfer"
)

const remoteBinPath = "/usr/local/bin/vastx"

func newShareCmd() *cobra.Command {
	var (
		ttl      time.Duration
		dest     string
		arch     string
		binary   string
		noBinary bool
		output   string
	)
	cmd := &cobra.Command{
		Use:   "share <name>",
		Short: "Create a credential-free download link and bootstrap command",
		Long: `Create a link that lets a GPU instance download a model without any storage
credentials. Rented machines are run by third parties, so keeping your R2
keys off them is the safer default.

The link points to a copy of the manifest whose file URLs are presigned. It
and every URL in it stop working after --ttl (at most 7 days).

Unless --no-binary is given, share also uploads the Linux vastx binary
(found next to this executable as vastx-linux-<arch>, or given with
--binary) and prints a one-line command that installs it and pulls the
model. Paste that into the instance shell or a vast.ai onstart script.

--output picks what goes to stdout: "text" (default, human readable), "cmd"
(only the bootstrap command) or "url" (only the share URL).`,
		Example: `  vastx share meta-llama/Llama-3.1-8B
  vastx share my-run/ckpt-2000 --ttl 6h --dest /workspace/ckpt --output cmd`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			name := args[0]
			if err := manifest.ValidName(name); err != nil {
				return err
			}
			switch output {
			case "text", "cmd", "url":
			default:
				return fmt.Errorf("--output must be text, cmd or url")
			}
			if dest == "" {
				dest = "/workspace/models/" + path.Base(name)
			}
			st, err := openStore(16)
			if err != nil {
				return err
			}
			m, err := st.GetManifest(ctx, name)
			if err != nil {
				return err
			}

			var binURL string
			if !noBinary {
				bin, err := findLinuxBinary(binary, arch)
				if err != nil {
					if output == "cmd" {
						return err
					}
					fmt.Fprintf(cmd.ErrOrStderr(), "warning: %v\n", err)
				} else if binURL, err = st.UploadBinary(ctx, bin, "vastx-linux-"+arch, ttl); err != nil {
					return err
				}
			}
			share, err := st.CreateShare(ctx, m, ttl)
			if err != nil {
				return err
			}

			pull := fmt.Sprintf("vastx pull %s %s", shellQuote(share.URL), shellQuote(dest))
			bootstrap := pull
			if binURL != "" {
				bootstrap = fmt.Sprintf("curl -fsSL -o %s %s && chmod +x %s && %s",
					remoteBinPath, shellQuote(binURL), remoteBinPath, pull)
			}
			out := cmd.OutOrStdout()
			switch output {
			case "url":
				fmt.Fprintln(out, share.URL)
			case "cmd":
				fmt.Fprintln(out, bootstrap)
			default:
				fmt.Fprintf(out, "share link for %s (%s, %s), expires %s:\n\n  %s\n\n",
					name, files(len(m.Files)), transfer.FormatBytes(m.TotalSize()), share.ExpiresAt.Format("2006-01-02 15:04 MST"), share.URL)
				if binURL != "" {
					fmt.Fprintf(out, "run on the instance:\n\n  %s\n", bootstrap)
				} else {
					fmt.Fprintf(out, "run on the instance (with vastx installed):\n\n  %s\n", pull)
				}
			}
			return nil
		},
	}
	f := cmd.Flags()
	f.DurationVar(&ttl, "ttl", 24*time.Hour, "how long the link works (max 168h)")
	f.StringVar(&dest, "dest", "", "download directory on the instance (default /workspace/models/<name>)")
	f.StringVar(&arch, "arch", "amd64", "instance CPU architecture (amd64 or arm64)")
	f.StringVar(&binary, "binary", "", "Linux vastx binary to upload (default: vastx-linux-<arch> next to this executable)")
	f.BoolVar(&noBinary, "no-binary", false, "don't upload a vastx binary")
	f.StringVarP(&output, "output", "o", "text", "what to print: text, cmd or url")
	return cmd
}

func findLinuxBinary(explicit, arch string) (string, error) {
	if arch != "amd64" && arch != "arm64" {
		return "", fmt.Errorf("--arch must be amd64 or arm64")
	}
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return "", err
		}
		return explicit, nil
	}
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if exe, err = filepath.EvalSymlinks(exe); err != nil {
		return "", err
	}
	if runtime.GOOS == "linux" && runtime.GOARCH == arch {
		return exe, nil
	}
	cand := filepath.Join(filepath.Dir(exe), "vastx-linux-"+arch)
	if _, err := os.Stat(cand); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("no Linux binary at %s; run `make install` or pass --binary", cand)
		}
		return "", err
	}
	return cand, nil
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
