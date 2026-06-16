import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import build_model, ResNet50
from utils import cutmix_data, set_seed


def test_resnet50_output_shape():
    model = build_model("resnet50", num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100), f"Expected (2, 100), got {out.shape}"


def test_se_resnet50_output_shape():
    model = build_model("se_resnet50", num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100), f"Expected (2, 100), got {out.shape}"


def test_se_resnet50_has_more_params():
    baseline = build_model("resnet50", num_classes=100)
    se_model = build_model("se_resnet50", num_classes=100)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    se_params = sum(p.numel() for p in se_model.parameters())
    assert se_params > baseline_params, "SE model should have more parameters"


def test_cutmix_preserves_shape():
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    x_mixed, y_a, y_b, lam = cutmix_data(x.clone(), y.clone(), alpha=1.0)
    assert x_mixed.shape == (4, 3, 32, 32), f"Shape mismatch: {x_mixed.shape}"
    assert y_a.shape == (4,)
    assert y_b.shape == (4,)


def test_cutmix_lambda_range():
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    for _ in range(50):
        _, _, _, lam = cutmix_data(x.clone(), y.clone(), alpha=1.0)
        assert 0.0 <= lam <= 1.0, f"Lambda out of range: {lam}"


def test_seed_reproducibility():
    set_seed(42)
    model1 = build_model("resnet50", num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    out1 = model1(x)

    set_seed(42)
    model2 = build_model("resnet50", num_classes=100)
    out2 = model2(x)

    assert torch.allclose(out1, out2), "Same seed should produce identical models"


def test_checkpoint_save_load(tmp_path=None):
    import tempfile
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()

    model = build_model("se_resnet50", num_classes=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    state = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": 75.5,
        "config": {"model": "se_resnet50", "epochs": 200},
    }

    ckpt_path = os.path.join(tmp_path, "test_ckpt.pth")
    torch.save(state, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu")
    assert loaded["epoch"] == 10
    assert loaded["best_val_acc"] == 75.5
    assert loaded["config"]["model"] == "se_resnet50"

    model2 = build_model("se_resnet50", num_classes=100)
    model2.load_state_dict(loaded["model_state_dict"])

    model.eval()
    model2.eval()
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(model(x), model2(x)), "Loaded model should produce same output"

    os.remove(ckpt_path)


def test_model_invalid_name():
    try:
        build_model("invalid_model", num_classes=100)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_resnet50_output_shape,
        test_se_resnet50_output_shape,
        test_se_resnet50_has_more_params,
        test_cutmix_preserves_shape,
        test_cutmix_lambda_range,
        test_seed_reproducibility,
        test_checkpoint_save_load,
        test_model_invalid_name,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
