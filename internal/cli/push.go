package cli

import (
	"fmt"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/ridoy/vastx/internal/transfer"
)

func newPushCmd() *cobra.Command {
	var (
		concurrency int
		chunkSize   string
		force       bool
		excludes    []string
	)
	cmd := &cobra.Command{
		Use:   "push <path> [name]",
		Short: "Upload a model directory or file to the bucket",
		Long: `Upload a model directory or file to the bucket under a name.

The name defaults to the base name of the path and may contain slashes,
e.g. meta-llama/Llama-3.1-8B. Pushing to an existing name updates it: files
whose size and modification time are unchanged are skipped, and files that
no longer exist locally are removed from the bucket. An interrupted push
resumes where it left off when run again.

Symlinks are followed, so a Hugging Face cache snapshot directory works.`,
		Example: `  vastx push ~/models/Llama-3.1-8B-Instruct
  vastx push ./out/checkpoint-2000 my-run/ckpt-2000`,
		Args: cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			src := args[0]
			name := ""
			if len(args) == 2 {
				name = args[1]
			} else {
				abs, err := filepath.Abs(src)
				if err != nil {
					return err
				}
				name = filepath.Base(abs)
			}
			cs, err := transfer.ParseSize(chunkSize)
			if err != nil {
				return err
			}
			st, err := openStore(concurrency)
			if err != nil {
				return err
			}
			res, err := transfer.Push(cmd.Context(), st, src, name, transfer.PushOptions{
				Concurrency:  concurrency,
				ChunkSize:    cs,
				Force:        force,
				Excludes:     append(append([]string{}, transfer.DefaultExcludes...), excludes...),
				Log:          cmd.ErrOrStderr(),
				ShowProgress: !quiet(cmd),
			})
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "pushed %s: %s, %s total; uploaded %s (%s) in %s",
				name, files(len(res.Manifest.Files)), transfer.FormatBytes(res.Manifest.TotalSize()),
				files(res.UploadedFiles), transfer.FormatBytes(res.UploadedBytes), res.Elapsed.Round(time.Second/10))
			if res.SkippedFiles > 0 {
				fmt.Fprintf(out, ", %d unchanged", res.SkippedFiles)
			}
			if res.DeletedObjects > 0 {
				fmt.Fprintf(out, ", removed %s deleted locally", files(res.DeletedObjects))
			}
			fmt.Fprintf(out, "\nnext: vastx share %s\n", name)
			return nil
		},
	}
	f := cmd.Flags()
	f.IntVarP(&concurrency, "concurrency", "j", transfer.DefaultPushConcurrency,
		"parts uploaded in parallel (memory use is concurrency x chunk size)")
	f.StringVar(&chunkSize, "chunk-size", "64MiB", "part size, at least 5MiB")
	f.BoolVar(&force, "force", false, "re-upload files even if they look unchanged")
	f.StringArrayVar(&excludes, "exclude", nil, "skip files or directories matching this glob (repeatable)")
	return cmd
}
