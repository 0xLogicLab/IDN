from idn.capabilities.base import Example
from idn.capabilities.cola import CoLASpec
from idn.capabilities.sst2 import SST2Spec
from idn.config import load_resolved_config


def _spec(path, cls):
    resolved = load_resolved_config(path)
    return cls(resolved.capability)


def test_sst2_parse_target_prompt_and_metric():
    spec = _spec("configs/capabilities/sst2.yaml", SST2Spec)
    ex = Example(
        id="1",
        prompt_inputs={"sentence": "A sharp and funny film."},
        reference=1,
    )

    assert spec.parse_prediction("Positive.") == 1
    assert spec.parse_prediction("The sentiment is negative") == 0
    assert spec.parse_prediction("unclear") is None
    assert spec.target_text(ex) == "positive"
    assert "positive or negative" in spec.format_prompt(ex)[1]["content"]
    assert spec.metric([1, 0, 1, None], [1, 0, 0, 1]) == {"accuracy": 0.5}


def test_cola_parse_target_prompt_and_metric():
    spec = _spec("configs/capabilities/cola.yaml", CoLASpec)
    ex = Example(
        id="2",
        prompt_inputs={"sentence": "The book was read by Kim."},
        reference=1,
    )

    assert spec.parse_prediction("Yes, acceptable.") == 1
    assert spec.parse_prediction("No.") == 0
    assert spec.parse_prediction("maybe") is None
    assert spec.target_text(ex) == "yes"
    assert "Answer yes or no" in spec.format_prompt(ex)[1]["content"]

    metrics = spec.metric([1, 0, 1, 0], [1, 0, 0, 1])
    assert metrics["accuracy"] == 0.5
    assert metrics["mcc"] == 0.0
