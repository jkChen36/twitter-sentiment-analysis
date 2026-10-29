"""单条文本情感预测演示：原始文本 -> 清洗分词 -> pad/truncate -> 模型 -> 情感概率"""
import argparse
import json
import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config as C
    from src.data_utils import Vocab, clean_text, pad_or_truncate, tokenize
    from src.models import build_model
else:
    from . import config as C
    from .data_utils import Vocab, clean_text, pad_or_truncate, tokenize
    from .models import build_model


def load_ckpt(ckpt_path, vocab_size):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = d["arch"]
    model = build_model(arch["rnn_type"], vocab_size, emb_dim=arch["emb_dim"],
                        hidden_dim=arch["hidden_dim"])
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model, arch


def predict(text, model, vocab, max_len=None):
    """单条文本推理：清洗分词 -> 截断/补齐 -> 模型 -> 三类概率"""
    max_len = max_len or C.MAX_LEN
    toks = tokenize(clean_text(text))
    ids, real_len = pad_or_truncate(vocab.encode(toks), max_len)
    X = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([max(real_len, 1)], dtype=torch.long)
    with torch.no_grad():
        logits = model(X, lengths)
        prob = torch.softmax(logits, dim=-1)[0]
    return prob.numpy(), toks, real_len


def best_ckpt():
    path = os.path.join(C.OUTPUTS, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    cands = []
    for grp in ("main", "fair", "w2v"):
        for v in m.get(grp, {}).values():
            cands.append((v["test"]["accuracy"], v.get("ckpt")))
    cands = [c for c in cands if c[1]]
    return max(cands)[1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="one vote can make all the difference anil kapoor "
                                      "answers modis election 2019 clarion call extends")
    ap.add_argument("--ckpt", default=None, help="模型权重路径，默认取测试集准确率最高的")
    ap.add_argument("--max-len", type=int, default=None)
    args = ap.parse_args()

    ckpt = args.ckpt or best_ckpt()
    if ckpt is None or not os.path.exists(ckpt):
        print("未找到模型权重，请先运行：python -m src.run_experiments")
        return

    vocab = Vocab.load(os.path.join(C.OUTPUTS, "vocab.json"))
    model, arch = load_ckpt(ckpt, len(vocab))

    print(f"模型：{arch['rnn_type'].upper()} | 向量化：{arch['scheme']} | L={args.max_len or C.MAX_LEN}")
    for t in [x for x in [args.text] if x]:
        prob, toks, n = predict(t, model, vocab, args.max_len)
        order = prob.argsort()[::-1]
        print(f"\n文本：{t}")
        print(f"分词（{n} 词）：{' '.join(toks)}")
        print("情感概率： " + "  ".join(f"{C.LABELS[i]}={prob[i]:.3f}" for i in order))
        print(f"预测：{C.LABELS[int(order[0])]}（置信度 {prob[order[0]]:.3f}）")


if __name__ == "__main__":
    main()
