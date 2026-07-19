import os

import torch

from model import build_model
from utils import cutmix_data, set_seed


def test_cutmix_preserves_shape():
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    x_mixed, y_a, y_b, lam = cutmix_data(x.clone(), y.clone(), alpha=1.0)
    assert x_mixed.shape == (4, 3, 32, 32)
    assert y_a.shape == (4,)
    assert y_b.shape == (4,)


def test_cutmix_lambda_range():
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    for _ in range(50):
        _, _, _, lam = cutmix_data(x.clone(), y.clone(), alpha=1.0)
        assert 0.0 <= lam <= 1.0


def test_seed_reproducibility():
    set_seed(42)
    model1 = build_model("resnet50", num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    out1 = model1(x)

    set_seed(42)
    model2 = build_model("resnet50", num_classes=100)
    out2 = model2(x)

    assert torch.allclose(out1, out2)


def test_checkpoint_save_load(tmp_path):
    model = build_model("se_resnet50", num_classes=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    state = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": 75.5,
        "config": {"model": "se_resnet50", "epochs": 200},
    }

    ckpt_path = os.path.join(str(tmp_path), "test_ckpt.pth")
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
    assert torch.allclose(model(x), model2(x))
