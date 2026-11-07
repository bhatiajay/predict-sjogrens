"""Project-wide settings: file locations, gene lists, and the random seed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "gse66795_subset.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = "model.keras"
META_FILE = "meta.json"

SEED = 42

# Illumina HumanHT-12 v4 probe IDs (GEO platform GPL10558) and the gene each one measures.
PROBE_TO_GENE = {
    "ILMN_2312606": "IRF5",
    "ILMN_2219628": "SSB",
    "ILMN_1678054": "TRIM21",
    "ILMN_1662358": "MX1",
    "ILMN_1760062": "IFI44",
    "ILMN_1707695": "IFIT1",
    "ILMN_1782704": "CD19",
    "ILMN_2066858": "TNFSF13B",
    "ILMN_1699651": "IL6",
    "ILMN_1718552": "CXCL13",
    "ILMN_2319077": "FAS",
    "ILMN_1690105": "STAT1",
    "ILMN_1807036": "AQP5",
}
GENE_TO_PROBE = {gene: probe for probe, gene in PROBE_TO_GENE.items()}
ALL_GENES = list(PROBE_TO_GENE.values())

# The five genes picked from published Sjögren's research for the original project.
# Choosing features from the literature (not from this dataset) keeps the evaluation honest.
REPORT_GENES = ["IRF5", "SSB", "TRIM21", "MX1", "IFI44"]

FEATURE_SETS = {"report": REPORT_GENES, "all": ALL_GENES}
