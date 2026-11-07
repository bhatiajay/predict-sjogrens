"""Plain-language descriptions of each gene, shown when you hover over or tap a gene name in the app.

Official names and IDs are from NCBI Gene (Homo sapiens). Descriptions stick to
well-established biology and say what the gene does, then why it's studied in
Sjögren's syndrome.
"""

GENE_INFO = {
    "IRF5": {
        "name": "Interferon regulatory factor 5",
        "aka": None,
        "group": "Interferon response",
        "ncbi_id": 3663,
        "does": "A switch protein inside immune cells. When a cell detects a virus, IRF5 turns on genes for interferons and other inflammatory signals.",
        "sjogrens": "Common inherited variants of IRF5 are among the best-replicated genetic risk factors for Sjögren's and lupus.",
    },
    "SSB": {
        "name": "La protein",
        "aka": "Official name: small RNA binding exonuclease protection factor La. The symbol SSB comes from its old name, Sjögren syndrome antigen B.",
        "group": "Autoantibody target",
        "ncbi_id": 6741,
        "does": "Binds newly made RNA in the nucleus and protects it while it is processed.",
        "sjogrens": "Many people with Sjögren's make antibodies against this protein (anti-La/SSB), one of the standard blood tests for the disease.",
    },
    "TRIM21": {
        "name": "Tripartite motif containing 21",
        "aka": "Also called Ro52.",
        "group": "Autoantibody target",
        "ncbi_id": 6737,
        "does": "Recognizes antibody-coated viruses that get inside a cell and marks them for destruction. Interferon increases how much of it cells make.",
        "sjogrens": "A main target of anti-Ro/SSA antibodies, the autoantibodies most often found in Sjögren's.",
    },
    "MX1": {
        "name": "MX dynamin like GTPase 1",
        "aka": "Its protein is called MxA.",
        "group": "Interferon response",
        "ncbi_id": 4599,
        "does": "An antiviral protein that blocks many viruses from copying themselves. Cells make it in response to type I interferon.",
        "sjogrens": "A classic marker of the type I interferon signature, a pattern of raised interferon-driven genes seen in many Sjögren's patients.",
    },
    "IFI44": {
        "name": "Interferon induced protein 44",
        "aka": None,
        "group": "Interferon response",
        "ncbi_id": 10561,
        "does": "Strongly switched on by type I interferon and linked to antiviral defense. Its exact function is still being worked out.",
        "sjogrens": "One of the most consistently raised genes in the interferon signature of Sjögren's and lupus.",
    },
    "IFIT1": {
        "name": "Interferon induced protein with tetratricopeptide repeats 1",
        "aka": "Also called ISG56.",
        "group": "Interferon response",
        "ncbi_id": 3434,
        "does": "Recognizes viral RNA that lacks the chemical markings of the cell's own RNA, and stops it from being used to make viral proteins.",
        "sjogrens": "Part of the type I interferon signature that is raised in many Sjögren's patients.",
    },
    "CD19": {
        "name": "CD19 molecule",
        "aka": None,
        "group": "B cells",
        "ncbi_id": 930,
        "does": "A protein on the surface of B cells, the immune cells that make antibodies. It helps them respond when they recognize a target.",
        "sjogrens": "B cells are overactive in Sjögren's and produce its autoantibodies. CD19 is a standard marker for counting them.",
    },
    "TNFSF13B": {
        "name": "TNF superfamily member 13b",
        "aka": "Also called BAFF or BLyS.",
        "group": "B cells",
        "ncbi_id": 10673,
        "does": "A signal that keeps B cells alive and helps them mature.",
        "sjogrens": "BAFF levels are often raised in Sjögren's, which may help self-reactive B cells survive. Drugs that block BAFF or its receptor have been tested in Sjögren's.",
    },
    "IL6": {
        "name": "Interleukin 6",
        "aka": None,
        "group": "Inflammation",
        "ncbi_id": 3569,
        "does": "A messenger protein (cytokine) that drives inflammation and fever and pushes B cells to make antibodies.",
        "sjogrens": "Often raised in autoimmune diseases, including in the blood and saliva of some Sjögren's patients.",
    },
    "CXCL13": {
        "name": "C-X-C motif chemokine ligand 13",
        "aka": "Also called BCA-1 or BLC.",
        "group": "B cells",
        "ncbi_id": 10563,
        "does": "A chemical signal that draws B cells into lymph nodes and other places where immune cells gather.",
        "sjogrens": "Helps form the clusters of immune cells that build up in the salivary glands in Sjögren's.",
    },
    "FAS": {
        "name": "Fas cell surface death receptor",
        "aka": "Also called CD95.",
        "group": "Cell death",
        "ncbi_id": 355,
        "does": "A receptor that, when triggered, tells a cell to self-destruct (apoptosis). The immune system uses it to remove cells it no longer needs.",
        "sjogrens": "Researchers have studied whether this kind of cell death in gland tissue contributes to the damage in Sjögren's.",
    },
    "STAT1": {
        "name": "Signal transducer and activator of transcription 1",
        "aka": None,
        "group": "Interferon response",
        "ncbi_id": 6772,
        "does": "Carries the interferon signal from the cell surface to the nucleus, where it switches on interferon-response genes.",
        "sjogrens": "Part of the interferon signature. Interferon raises STAT1 itself too, which amplifies the response.",
    },
    "AQP5": {
        "name": "Aquaporin 5",
        "aka": None,
        "group": "Gland function",
        "ncbi_id": 362,
        "does": "A water channel in the cells of salivary and tear glands that lets water through to form saliva and tears.",
        "sjogrens": "Its placement in gland cells has been reported to be disrupted in Sjögren's. It's mostly made in glands, not blood, which may be why it barely differs in these blood samples.",
    },
}


def ncbi_url(gene: str) -> str:
    return f"https://www.ncbi.nlm.nih.gov/gene/{GENE_INFO[gene]['ncbi_id']}"
