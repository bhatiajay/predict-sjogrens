from sjogrens import config
from sjogrens.genes import GENE_INFO, ncbi_url

FIELDS = {"name", "aka", "group", "ncbi_id", "does", "sjogrens"}


def test_every_gene_has_an_explanation():
    assert set(GENE_INFO) == set(config.ALL_GENES)
    for gene, info in GENE_INFO.items():
        assert set(info) == FIELDS, gene
        for field in ("name", "group", "does", "sjogrens"):
            assert info[field].strip(), f"{gene}.{field} is empty"
        assert isinstance(info["ncbi_id"], int)


def test_ncbi_links_point_at_gene_pages():
    assert ncbi_url("MX1") == "https://www.ncbi.nlm.nih.gov/gene/4599"
