# Job templates

Every script takes its paths from `configs/paths.yaml`, so these templates set
one variable and nothing else needs editing:

```bash
export ANTIGEN_EMBEDDING_ROOT=/mnt/bioadhoc/Groups/Peters/Self-similarity
```

That replaces the `path = '/mnt/bioadhoc/...'` line that used to sit at the top
of all eleven scripts. Submit with `sbatch slurm/<file>.sh`; override anything
else on the command line with `--variant` or `--set key=value`.

| file | what it runs | needs a GPU |
|---|---|---|
| `embed_esmc.sh` | ESMC-300M over proteins and peptides | yes |
| `embed_protbert.sh` | ProtBert over proteins and peptides | yes |
| `embed_peptidebert.sh` | the three fine-tuned PeptideBERT checkpoints | yes |
| `umap.sh` | the UMAP runner for one variant | no |
| `separability.sh` | silhouette / PERMANOVA / Fisher | no |

The embedders checkpoint atomically and resume, so a job that hits its time
limit can simply be resubmitted.
