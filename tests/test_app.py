import re

import pytest

from app import create_app
from sjogrens import artifacts
from sjogrens.data import load_dataset
from sjogrens.model import train

GENES = ["IRF5", "SSB", "TRIM21", "MX1", "IFI44"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Train a quick model into a temp folder, the same way train.py saves one."""
    X, y = load_dataset(genes=GENES)
    out = tmp_path_factory.mktemp("models")
    artifacts.save(train(X, y, epochs=30), artifacts.build_meta(X, y, GENES, cv_summary=None), out)
    return out


@pytest.fixture(scope="module")
def client(model_dir):
    return create_app(model_dir).test_client()


@pytest.fixture(scope="module")
def meta(model_dir):
    return artifacts.load(model_dir)[1]


def score_of(response) -> float:
    match = re.search(r'data-score="([0-9.]+)"', response.get_data(as_text=True))
    assert match, "result page has no score"
    return float(match.group(1))


def test_home_page_has_a_field_per_gene(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for gene in GENES:
        assert f'name="{gene}"' in html


def test_example_link_prefills_values(client, meta):
    html = client.get("/?example=patient").get_data(as_text=True)
    assert f'value="{meta["examples"]["patient"]["MX1"]}"' in html


def test_valid_input_returns_a_score(client, meta):
    response = client.post("/predict", data=meta["examples"]["control"])
    assert response.status_code == 200
    assert 0.0 <= score_of(response) <= 1.0


def test_typical_patient_outscores_typical_control(client, meta):
    patient = score_of(client.post("/predict", data=meta["examples"]["patient"]))
    control = score_of(client.post("/predict", data=meta["examples"]["control"]))
    assert patient > control


def test_blank_field_is_rejected(client, meta):
    data = {**meta["examples"]["patient"], "MX1": ""}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert "Required." in response.get_data(as_text=True)


def test_non_numeric_field_is_rejected(client, meta):
    data = {**meta["examples"]["patient"], "IFI44": "abc"}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert "isn&#39;t a number" in response.get_data(as_text=True)


def test_infinite_value_is_rejected(client, meta):
    data = {**meta["examples"]["patient"], "SSB": "inf"}
    assert client.post("/predict", data=data).status_code == 400


def test_out_of_range_value_warns(client, meta):
    data = {**meta["examples"]["patient"], "MX1": "25"}
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert "Outside the training data" in response.get_data(as_text=True)


def test_missing_model_shows_instructions(tmp_path):
    client = create_app(tmp_path).test_client()
    response = client.get("/")
    assert response.status_code == 503
    assert "uv run python train.py" in response.get_data(as_text=True)
    assert client.post("/predict", data={}).status_code == 503
