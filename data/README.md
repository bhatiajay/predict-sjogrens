# Data

`gse66795_subset.csv` is a subset of the public GEO series
[GSE66795](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66795):
*"A transcriptional signature of fatigue derived from patients with primary Sjögren's Syndrome"*.

- **Samples:** 160 whole-blood samples (GSM1632237–GSM1632396): 131 primary Sjögren's
  syndrome patients and 29 healthy controls. All are female.
- **Platform:** Illumina HumanHT-12 v4 expression BeadChip
  ([GPL10558](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10558)).
- **Values:** normalized log2 intensities from the series matrix, as published.
- **Subset:** 13 probes for genes linked to Sjögren's syndrome in the literature.

## Columns

| Column | Meaning |
|---|---|
| `GEO Accession #` | GEO sample ID |
| `Patient Group` | `patient group: Patient` (label 1) or `patient group: Control` (label 0) |
| `Gender` | always `gender: Female` in this subset, so it's not used as a feature |
| `ILMN_…` | expression of one probe (see below) |

| Probe | Gene | Used by default model |
|---|---|---|
| ILMN_2312606 | IRF5 | ✓ |
| ILMN_2219628 | SSB | ✓ |
| ILMN_1678054 | TRIM21 | ✓ |
| ILMN_1662358 | MX1 | ✓ |
| ILMN_1760062 | IFI44 | ✓ |
| ILMN_1707695 | IFIT1 | |
| ILMN_1782704 | CD19 | |
| ILMN_2066858 | TNFSF13B | |
| ILMN_1699651 | IL6 | |
| ILMN_1718552 | CXCL13 | |
| ILMN_2319077 | FAS | |
| ILMN_1690105 | STAT1 | |
| ILMN_1807036 | AQP5 | |

The probe-to-gene mapping lives in `sjogrens/config.py`.

## Note on the original write-up

The original science-fair report named GSE51092 as the training data. This CSV is
actually GSE66795: the sample IDs above belong to that series. GSE51092 is a separate
Sjögren's cohort (190 patients, 32 controls) on a different Illumina chip (GPL6884), and
it doesn't include the IRF5 or SSB probes.
