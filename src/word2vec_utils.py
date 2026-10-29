"""任务 2 - 方式 B：使用 gensim 训练 Word2Vec 预训练词向量
   优先调用 gensim；若环境未安装 gensim，自动回退到内置 SGNS（Skip-gram + 负采样）实现，
   保证实验在无网络/无第三方依赖时仍可完整复现。
"""
import os
import random

import numpy as np

from .config import (DATA_PROC, W2V_DIM, W2V_EPOCHS, W2V_MIN_COUNT, W2V_NEGATIVE,
                     W2V_WINDOW)

W2V_CACHE = os.path.join(DATA_PROC, "w2v_matrix.npy")


def _has_gensim():
    try:
        import gensim  # noqa: F401
        return True
    except Exception:
        return False


def _backend_tag():
    return "gensim" if _has_gensim() else "内建 SGNS"


class _SGNS:
    """纯 numpy 的 Skip-gram + Negative Sampling（gensim 缺失时的等价替代）"""

    def __init__(self, vocab_size, dim, seed=42):
        rng = np.random.default_rng(seed)
        limit = 0.5 / dim
        self.V = rng.uniform(-limit, limit, (vocab_size, dim)).astype(np.float32)   # 中心词
        self.U = rng.uniform(-limit, limit, (vocab_size, dim)).astype(np.float32)   # 上下文词
        self.rng = rng

    @staticmethod
    def _sigmoid(x):
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    def _make_pairs(self, corpus_ids, window, max_pairs=8_000_000):
        centers, contexts = [], []
        rnd = random.Random(1234)
        for ids in corpus_ids:
            n = len(ids)
            if n < 2:
                continue
            for i, c in enumerate(ids):
                w = rnd.randint(1, window)
                lo, hi = max(0, i - w), min(n, i + w + 1)
                for j in range(lo, hi):
                    if j != i:
                        centers.append(c)
                        contexts.append(ids[j])
        idx = self.rng.permutation(len(centers))[:max_pairs]
        return np.asarray(centers, dtype=np.int64)[idx], np.asarray(contexts, dtype=np.int64)[idx]

    def train(self, corpus_ids, window=W2V_WINDOW, dim=W2V_DIM, epochs=W2V_EPOCHS,
              neg=W2V_NEGATIVE, lr=0.05, batch=2048, vocab_size=None, freq=None):
        centers, contexts = self._make_pairs(corpus_ids, window)
        n = len(centers)
        # 负采样分布：unigram^3/4
        if freq is None:
            prob = np.ones(vocab_size, dtype=np.float64)
        else:
            prob = np.power(np.asarray(freq, dtype=np.float64), 0.75)
        prob = prob / prob.sum()

        total_steps = int(np.ceil(n / batch)) * epochs
        step = 0
        for ep in range(epochs):
            order = self.rng.permutation(n)
            for s in range(0, n, batch):
                b = order[s:s + batch]
                if len(b) == 0:
                    continue
                step += 1
                alpha = lr * max(1e-4, 1.0 - step / max(1, total_steps))
                vc = self.V[centers[b]]                     # (B, D)
                up = self.U[contexts[b]]                    # (B, D)
                neg_ids = self.rng.choice(vocab_size, size=(len(b), neg), p=prob)
                un = self.U[neg_ids]                        # (B, K, D)

                sp = np.einsum("bd,bd->b", vc, up)
                sn = np.einsum("bd,bkd->bk", vc, un)
                gp = self._sigmoid(sp) - 1.0                # (B,)
                gn = self._sigmoid(sn)                      # (B, K)

                grad_vc = gp[:, None] * up + np.einsum("bk,bkd->bd", gn, un)
                grad_up = gp[:, None] * vc
                grad_un = gn[:, :, None] * vc[:, None, :]

                self.V[centers[b]] -= alpha * grad_vc
                self.U[contexts[b]] -= alpha * grad_up
                np.add.at(self.U, neg_ids, -alpha * grad_un)
        return self.V


def train_word2vec(token_lists, vocab, dim=W2V_DIM, cache_path=None, use_cache=True):
    """返回形状 (vocab_size, dim) 的预训练词向量矩阵，行序与词表一致

    行序必须和词表一致，才能直接塞进 nn.Embedding 当初始值。
    训练一次要几分钟，所以结果缓存到 data/processed，第二次跑直接读缓存。
    """
    if cache_path is None:
        cache_path = os.path.join(DATA_PROC, f"w2v_matrix_{dim}.npy")
    if use_cache and os.path.exists(cache_path):
        mat = np.load(cache_path)
        if mat.shape == (len(vocab), dim):
            return mat.astype(np.float32), _backend_tag() + "(缓存)"

    corpus_ids = [[vocab.stoi.get(t, 1) for t in toks] for toks in token_lists]

    if _has_gensim():
        from gensim.models import Word2Vec
        model = Word2Vec(sentences=token_lists, vector_size=dim, window=W2V_WINDOW,
                         min_count=W2V_MIN_COUNT, sg=1, negative=W2V_NEGATIVE,
                         epochs=W2V_EPOCHS, workers=4, seed=42)
        matrix = np.zeros((len(vocab), dim), dtype=np.float32)
        hit = 0
        for w, i in vocab.stoi.items():
            if w in model.wv:
                matrix[i] = model.wv[w]
                hit += 1
        # OOV（<PAD>/<UNK>/低频词）：使用语料向量的小随机初始化（与训练同尺度）
        rng = np.random.default_rng(0)
        std = float(np.std(matrix[hit > 0])) if hit else 0.1
        std = std if std > 0 else 0.1
        miss = [i for w, i in vocab.stoi.items() if w not in model.wv]
        if miss:
            matrix[miss] = rng.normal(0, std * 0.5, (len(miss), dim)).astype(np.float32)
        backend = _backend_tag()
    else:
        freq = np.zeros(len(vocab), dtype=np.float64)
        for ids in corpus_ids:
            for i in ids:
                freq[i] += 1
        sgns = _SGNS(len(vocab), dim)
        matrix = sgns.train(corpus_ids, dim=dim, vocab_size=len(vocab), freq=freq).astype(np.float32)
        backend = "内建 SGNS（Skip-gram + 负采样）"

    matrix[0] = 0.0   # <PAD> 向量固定为 0
    np.save(cache_path, matrix)
    return matrix, backend


def backend_name():
    return "gensim" if _has_gensim() else "内建 SGNS（Skip-gram + 负采样）"
