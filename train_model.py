#!/usr/bin/env python3
"""使用 PyTorch 训练/导出用于 GUI 的手写数字识别模型。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

ARTIFACT_DIR = Path("artifacts")
SCRIPTED_MODEL_PATH = Path("model_scripted.pt")
STATE_MODEL_PATH = Path("model_state.pt")
HISTORY_PATH = ARTIFACT_DIR / "training_history.json"
CURVE_PATH = ARTIFACT_DIR / "training_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18, help="训练轮数（默认：18）")
    parser.add_argument("--batch-size", type=int, default=192, help="批大小（默认：192）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（默认：1e-3）")
    parser.add_argument("--val-split", type=float, default=0.12, help="验证集比例（默认：0.12）")
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层宽度（默认：128）")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数（默认：4）")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="MNIST 数据存放目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    parser.add_argument("--no-plot", action="store_true", help="训练结束后不显示曲线")
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=CURVE_PATH,
        help="保存训练曲线的路径（默认 artifacts/training_curves.png）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="手动指定训练设备（cpu / cuda / mps）。默认自动检测。",
    )
    return parser.parse_args()


class Net(nn.Module):
    """轻量级 CNN + MLP 分类头，相比全连接网络有更高准确率。"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Dropout(0.05),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)
        return torch.log_softmax(logits, dim=1)


def _build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), shear=6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return train_transform, eval_transform


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _build_dataloaders(
    batch_size: int,
    val_split: float,
    num_workers: int,
    data_dir: Path,
    seed: int,
) -> DataLoaders:
    train_tf, eval_tf = _build_transforms()
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    val_base = datasets.MNIST(data_dir, train=True, download=False, transform=eval_tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=eval_tf)

    val_size = int(len(full_train) * val_split)
    if not 0 < val_size < len(full_train):
        raise ValueError("val_split 需要在 (0,1) 范围内。")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_base, val_indices)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return DataLoaders(train_loader, val_loader, test_loader)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        total_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def _export_model(model: nn.Module) -> None:
    SCRIPTED_MODEL_PATH.parent.mkdir(exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    scripted.save(SCRIPTED_MODEL_PATH.as_posix())
    torch.save(model.state_dict(), STATE_MODEL_PATH)
    print(f"模型已保存：\n  - TorchScript: {SCRIPTED_MODEL_PATH}\n  - State Dict : {STATE_MODEL_PATH}")


def _save_history(history: Dict[str, list]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, ensure_ascii=False, indent=2)
    print(f"训练日志已保存至: {HISTORY_PATH}")


def _plot_history(history: Dict[str, list], save_path: Path | None, show_plot: bool) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="训练损失")
    plt.plot(epochs, history["val_loss"], label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="训练准确率")
    plt.plot(epochs, history["val_acc"], label="验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"训练曲线已保存至: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def _pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _pick_device(args.device)
    print(f"使用设备: {device}")

    loaders = _build_dataloaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    model = Net(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_kwargs = dict(mode="max", factor=0.6, patience=2, min_lr=1e-5)
    if "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
        scheduler_kwargs["verbose"] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, loaders.train, optimizer, device)
        train_metrics = _evaluate(model, loaders.train, device)
        val_metrics = _evaluate(model, loaders.val, device)
        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_metrics['accuracy']:.4%} | "
            f"val loss: {val_metrics['loss']:.4f}, val acc: {val_metrics['accuracy']:.4%}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"加载验证集最佳权重，val_acc={best_val_acc:.4%}")

    test_metrics = _evaluate(model, loaders.test, device)
    print(f"测试集准确率: {test_metrics['accuracy']:.4%}, 损失: {test_metrics['loss']:.4f}")

    _export_model(model)
    _save_history(history)
    _plot_history(history, save_path=args.save_plot, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
