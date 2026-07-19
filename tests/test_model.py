import pytest
import torch

from model import build_model


@pytest.mark.parametrize("model_name", ["resnet50", "se_resnet50"])
def test_model_output_shape(model_name):
    model = build_model(model_name, num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_se_resnet50_has_more_params_than_baseline():
    baseline = build_model("resnet50", num_classes=100)
    se_model = build_model("se_resnet50", num_classes=100)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    se_params = sum(p.numel() for p in se_model.parameters())
    assert se_params > baseline_params


def test_model_invalid_name_raises_value_error():
    with pytest.raises(ValueError):
        build_model("invalid_model", num_classes=100)
