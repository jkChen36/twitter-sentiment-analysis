"""任务 1：数据预处理
原始文本 -> 清洗 -> 分词 -> 数值化 -> padding + truncate 统一长度 L -> N x L -> 划分 train/val/test
"""
import csv
import json
import os
import random
import re
from collections import Counter

import numpy as np

from . import config as C
from .config import (LABEL_MAP, LABELS, MIN_FREQ, PAD_IDX, PAD_TOKEN, SEED,
                     SPLIT_RATIO, UNK_IDX, UNK_TOKEN, TRAIN_CSV, VALID_CSV)

csv.field_size_limit(10 ** 7)

# URL / @用户 / 非字母数字字符
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Twitter 文本清洗：去 URL、去 @用户、小写化、只保留字母数字与空格。

    和题目给的样例风格一致：
      'one vote can make all the difference anil kapoor answers modis election 2019 clarion call extends'
    """
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.lower()
    text = text.replace("#", " ")          # '#' 仅作为分隔符，话题词保留
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    return text.split()


def load_raw_csv(path):
    """读取原始 csv（无表头，列：id, entity, sentiment, content）"""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            _id, entity, sentiment, content = row[0], row[1], row[2], ",".join(row[3:])
            sentiment = sentiment.strip()
            if sentiment not in LABEL_MAP:
                continue
            rows.append((entity.strip(), sentiment, content.strip()))
    return rows


def load_dataset():
    """合并 twitter_training.csv 与 twitter_validation.csv"""
    rows = load_raw_csv(TRAIN_CSV)
    if os.path.exists(VALID_CSV):
        rows += load_raw_csv(VALID_CSV)

    if C.MAX_SAMPLES is not None and len(rows) > C.MAX_SAMPLES:
        random.Random(SEED).shuffle(rows)
        rows = rows[:C.MAX_SAMPLES]

    texts, labels = [], []
    for _ent, sent, content in rows:
        cleaned = clean_text(content)
        if len(cleaned) == 0:
            continue
        texts.append(cleaned)
        labels.append(LABEL_MAP[sent])
    return texts, np.array(labels, dtype=np.int64)


# ------------------------------ 词表 ------------------------------
class Vocab:
    """词表：词 <-> 整数 ID 的双向映射

    索引 0 固定给 <PAD>（补齐标记），索引 1 给 <UNK>（未登录词/低频词）。
    词频低于 min_freq 的词不会进词表，编码时统一落到 <UNK>。
    """

    def __init__(self, tokens_list, min_freq=MIN_FREQ):
        counter = Counter()
        for toks in tokens_list:
            counter.update(toks)
        self.itos = [PAD_TOKEN, UNK_TOKEN]
        for w, c in counter.most_common():
            if c >= min_freq:
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.min_freq = min_freq

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "min_freq": self.min_freq}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = cls.__new__(cls)
        v.itos = obj["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        v.min_freq = obj.get("min_freq", MIN_FREQ)
        return v


def pad_or_truncate(seq, max_len=None, pad_idx=PAD_IDX):
    """统一句子长度：长度 > L 截断丢弃尾部；长度 < L 用 <PAD> 补齐至 L。

    返回 (定长序列, 真实长度)。真实长度要留给后面的 pack_padded_sequence 用，
    循环层靠它知道每个样本该读到第几个词为止。
    """
    max_len = max_len or C.MAX_LEN
    if len(seq) > max_len:
        # truncate：只保留前 L 个词，尾部多余内容丢弃
        return seq[:max_len], max_len
    # padding：尾部补 pad_idx（0），真实长度仍然是原来的词数
    return seq + [pad_idx] * (max_len - len(seq)), len(seq)


def encode_all(token_lists, vocab, max_len=None):
    """输出形状 N x L 的整数序列矩阵，以及每条样本的真实长度（供 mask 使用）"""
    max_len = max_len or C.MAX_LEN
    n = len(token_lists)
    data = np.full((n, max_len), PAD_IDX, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = vocab.encode(toks)
        padded, l = pad_or_truncate(ids, max_len)
        data[i] = padded
        lengths[i] = l
    lengths = np.clip(lengths, 1, None)     # 避免除零 / 空序列
    return data, lengths


def stratified_split(labels, ratios=SPLIT_RATIO, seed=SEED):
    """按类别比例划分 train / val / test，返回三段索引

    分层的意思是每个类别内部各自按 8:1:1 切，再拼起来，
    这样三段的类别分布和整体一致，不会因为随机划分导致某一类在测试集里偏少。
    """
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(labels == c)[0] for c in range(len(LABELS))]
    for arr in idx_by_class:
        rng.shuffle(arr)

    train_idx, val_idx, test_idx = [], [], []
    for arr in idx_by_class:
        n = len(arr)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        train_idx.append(arr[:n_tr])
        val_idx.append(arr[n_tr:n_tr + n_va])
        test_idx.append(arr[n_tr + n_va:])

    def merge(parts):
        out = np.concatenate(parts)
        rng.shuffle(out)
        return out

    return merge(train_idx), merge(val_idx), merge(test_idx)


def build_dataset(max_len=None, min_freq=MIN_FREQ):
    """端到端预处理流水线

    清洗分词 -> 建词表 -> 编码 + 截断/补齐（得到 N x L） -> 分层划分三段
    返回的 X 形状为 (N, L)，lengths 记录每条样本的真实词数。
    """
    max_len = max_len or C.MAX_LEN
    texts, labels = load_dataset()
    token_lists = [tokenize(t) for t in texts]

    vocab = Vocab(token_lists, min_freq=min_freq)
    data, lengths = encode_all(token_lists, vocab, max_len=max_len)

    tr, va, te = stratified_split(labels)
    raw_lens = np.array([len(t) for t in token_lists], dtype=np.int64)

    dataset = {
        "X": data, "y": labels, "lengths": lengths,
        "vocab": vocab, "token_lists": token_lists,
        "raw_token_lengths": raw_lens,
        "split": {
            "train": {"X": data[tr], "y": labels[tr], "lengths": lengths[tr]},
            "val": {"X": data[va], "y": labels[va], "lengths": lengths[va]},
            "test": {"X": data[te], "y": labels[te], "lengths": lengths[te]},
        },
    }
    return dataset


def describe_dataset(dataset):
    """预处理阶段的可视化统计信息"""
    raw = dataset["raw_token_lengths"]
    stats = {
        "num_samples": len(dataset["y"]),
        "max_len_L": dataset["X"].shape[1],
        "shape": tuple(dataset["X"].shape),
        "vocab_size": len(dataset["vocab"]),
        "raw_len_mean": float(raw.mean()),
        "raw_len_p50": float(np.percentile(raw, 50)),
        "raw_len_p90": float(np.percentile(raw, 90)),
        "raw_len_p95": float(np.percentile(raw, 95)),
        "raw_len_max": int(raw.max()),
        "truncated_ratio": float((raw > dataset["X"].shape[1]).mean()),
        "padded_ratio": float((raw < dataset["X"].shape[1]).mean()),
        "class_counts": {LABELS[i]: int((dataset["y"] == i).sum()) for i in range(len(LABELS))},
        "split_sizes": {k: len(v["y"]) for k, v in dataset["split"].items()},
    }
    return stats
