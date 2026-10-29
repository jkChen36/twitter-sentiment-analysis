"""任务 4 + 任务 5：训练框架（交叉熵损失）与评估指标"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (BATCH_SIZE, DEVICE, GRAD_CLIP, LR, NUM_EPOCHS, PATIENCE,
                     WEIGHT_DECAY)


# ------------------------------ 指标 ------------------------------
def compute_metrics(y_true, y_pred, num_classes=3):
    """算 Accuracy、Macro-F1、各类别 P/R/F1 与混淆矩阵

    Macro-F1 是三类 F1 的算术平均，不按样本数加权，
    所以样本少的那一类和多的那一类权重一样，类别不均衡时更能反映真实水平。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class, f1s = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class.append({"precision": prec, "recall": rec, "f1": f1, "support": int(cm[c, :].sum())})
        f1s.append(f1)
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------- DataLoader ----------------------------
def make_loaders(split, batch_size=BATCH_SIZE, num_workers=0):
    """split: dict with X / y / lengths"""
    def to_ds(part):
        X = torch.as_tensor(part["X"], dtype=torch.long)
        y = torch.as_tensor(part["y"], dtype=torch.long)
        L = torch.as_tensor(part["lengths"], dtype=torch.long)
        return TensorDataset(X, y, L)

    g = torch.Generator()
    g.manual_seed(0)
    loaders = {}
    for name, part in split.items():
        shuffle = (name == "train")
        loaders[name] = DataLoader(to_ds(part), batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_workers, generator=g,
                                   drop_last=False, pin_memory=(DEVICE == "cuda"))
    return loaders


# ------------------------------ 训练 ------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for X, y, lengths in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X, lengths)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train_model(model, loaders, device=DEVICE, epochs=NUM_EPOCHS, lr=LR,
                weight_decay=WEIGHT_DECAY, patience=PATIENCE, grad_clip=GRAD_CLIP,
                verbose=True, tag=""):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "epoch_time": []}
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    bad = 0
    t_start = time.time()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, nb = 0.0, 0
        for X, y, lengths in loaders["train"]:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X, lengths)                  # 前向：序列 + 真实长度
            loss = criterion(logits, y)                 # 交叉熵
            loss.backward()
            if grad_clip:
                # 梯度裁剪：RNN 容易梯度爆炸，按范数截断到 5.0
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * X.size(0)
            nb += X.size(0)

        val_m = evaluate(model, loaders["val"], criterion, device)
        history["train_loss"].append(running / max(nb, 1))
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["epoch_time"].append(time.time() - t0)

        if verbose:
            print(f"  [{tag}] epoch {ep:02d} | train_loss {history['train_loss'][-1]:.4f} "
                  f"| val_loss {val_m['loss']:.4f} | val_acc {val_m['accuracy']:.4f} "
                  f"| val_macroF1 {val_m['macro_f1']:.4f} | {history['epoch_time'][-1]:.1f}s",
                  flush=True)

        # 按验证集准确率保存最优权重：验证集只用来挑轮次和早停，不参与梯度更新
        if val_m["accuracy"] > best["val_acc"]:
            best.update(val_acc=val_m["accuracy"], epoch=ep,
                        state=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [{tag}] early stop at epoch {ep}", flush=True)
                break

    # 回退到验证集上表现最好的那一份权重，再去测测试集
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # 测试集全程只在这里被读一次，不参与训练也不参与模型选择
    test_m = evaluate(model, loaders["test"], criterion, device)
    history["total_time"] = time.time() - t_start
    history["best_epoch"] = best["epoch"]
    return history, test_m, model
