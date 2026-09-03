# Data

Peptides tested for T cell responses, retrieved from **IEDB** on **11 September 2025**,
plus the source antigen sequences they derive from and NCBI taxonomy annotation.

All tables are gzipped CSV. `pandas.read_csv` reads them directly:

```python
epi  = pd.read_csv('data/epitopes.csv.gz')
prot = pd.read_csv('data/proteins.csv.gz')
df   = epi.merge(prot, on='protein_id')          # peptide + its parent protein
```

## Retrieval and filters

Query filters applied at retrieval:

- Linear peptides only
- No post-translational modifications
- HLA of high resolution
- Tested in humans
- References from 2010 onward
- A `curated_source_antigen` exists, and the epitope is found within the retrieved protein sequence

A peptide–MHC pair derived from a specific source protein is labelled **positive**
(`immune_response = 1`) if at least one assay was reported positive.

## Tables

### `epitopes.csv.gz` — 30,379 rows
One row per peptide–MHC–antigen combination. The modelling table.

| column | notes |
|---|---|
| `linear_sequence` | the peptide |
| `linear_sequence_length` | peptide length in residues |
| `mhc_class` | I or II |
| `mhc_restriction` | HLA allele, high resolution |
| `curated_source_antigen_start` | 1-based, inclusive, into the parent protein |
| `curated_source_antigen_end` | 1-based, inclusive |
| `immune_response` | 1 = positive in ≥1 assay, 0 = negative |
| `protein_id` | foreign key → `proteins.csv.gz` |
| `coord_ok` | 1 if `protein_sequence[start-1:end] == linear_sequence`, else 0 |

**`coord_ok` matters.** 13 of 30,379 rows have coordinates that do not locate the
peptide in the stated protein. All 13 peptides *are* present in their protein at a
different offset — five of them off by exactly −14 on the same 658-residue protein,
which suggests a signal-peptide numbering difference upstream. Because the embedding
pipeline slices `protein_embedding[start:end]`, those rows would embed the wrong
residues. Filter on `coord_ok == 1` before slicing. Details and suggested corrections
are in `results/qc/coordinate_mismatches.csv`; nothing has been silently corrected here.

### `proteins.csv.gz` — 4,966 rows
Unique source antigen sequences, keyed by content hash.

| column | notes |
|---|---|
| `protein_id` | first 12 hex of `sha1(protein_sequence)` — stable across regenerations |
| `protein_length` | residues |
| `protein_sequence` | full amino-acid sequence |

### `antigens.csv.gz` — 5,295 rows
Maps IEDB antigen identifiers onto proteins. Many-to-one: 5,295 IRIs resolve to
4,966 distinct sequences, and 17 IRIs have no sequence (blank `protein_id`).

| column | notes |
|---|---|
| `curated_source_antigen_iri` | e.g. `GENPEPT:NP_001070958.1` |
| `protein_id` | foreign key → `proteins.csv.gz`, blank where no sequence was retrieved |

### `epitopes_taxonomy.csv.gz` — 58,174 rows
The ungrouped assay-level table with NCBI taxonomy resolved to 12 ranks
(`tax_domain` … `tax_most_specific`). This is the input to the downstream UMAP and
separability analyses. Joins to `epitopes.csv.gz` on
`[linear_sequence, mhc_class, mhc_restriction, curated_source_antigen_start, curated_source_antigen_end]`.

Also carries `source_organism_name`, `disease_names`, `assay_method` and
`assay_response_measured` for stratification.

### `epitopes_annotated.csv.gz` — 58,174 rows
The same ungrouped table *before* taxonomy resolution. Kept for provenance; use
`epitopes_taxonomy.csv.gz` unless you specifically need the pre-annotation state.

### `subset.csv` — 599 rows
A small dummy set for pipeline development, left uncompressed so the schema is
readable on GitHub. **Note its schema differs from the tables above**
(`sequence`, `uniprot_id`, `type`, `species`, `immune_response`, `protein_sequence`) —
it predates the current pull and is not a subset of `epitopes.csv.gz`.

### `disease_ontology.csv` — 6 rows
Disease category → DOID mapping used to group `disease_names`.

## Quality control

`results/qc/` holds diagnostics, not pipeline inputs:

- `coordinate_mismatches.csv` — the 13 rows above, with occurrence counts and suggested offsets
- `duplicate_epitopes.csv.gz` — 1,684 duplicate rows (1,207 distinct peptides)
- `duplicate_epitopes_annotated.csv.gz` — 26,714 duplicate rows at assay level

## Provenance and licence

Derived from the **Immune Epitope Database (IEDB)** and **CEDAR**, retrieved
11 September 2025 via their public APIs. IEDB content is released under
**CC BY 4.0**; this derived dataset is redistributed under the same terms.
Please cite IEDB if you use it.

Storage note: `epitopes.csv.gz` stores `protein_id` rather than the sequence itself.
The previous flat file repeated each protein across every one of its peptides —
48.0 MB of 49.5 MB — and carried no key back to the protein table.
