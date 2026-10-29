"""任务 3 + 任务 4：循环神经网络特征提取器 + MLP 分类头
单向：RNN / LSTM / GRU        双向：BiRNN / BiLSTM / BiGRU
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import (DROPOUT, EMB_DIM, HIDDEN_DIM, MLP_HIDDEN, NUM_LAYERS, PAD_IDX,
                     POOLING)

RNN_REGISTRY = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}


def _make_rnn(rnn_type, input_dim, hidden_dim, bidirectional, num_layers, dropout):
    """按名字取循环层。名字里的 "bi" 只用来决定 bidirectional，
    真正选单元类型之前会先把它去掉，所以 birnn 和 rnn 拿到的是同一个 nn.RNN。
    """
    key = rnn_type.lower().replace("bi", "")
    if key not in RNN_REGISTRY:
        raise ValueError(f"不支持的循环单元：{rnn_type}")
    cls = RNN_REGISTRY[key]
    return cls(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
               bidirectional=bidirectional, batch_first=True,
               dropout=dropout if num_layers > 1 else 0.0)


class RecSentimentClassifier(nn.Module):
    """Embedding(或 Word2Vec) -> 循环特征提取器 -> MLP 分类头 -> 3 类情感 logits"""

    def __init__(self,
                 vocab_size: int,
                 num_classes: int = 3,
                 emb_dim: int = EMB_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 rnn_type: str = "lstm",
                 bidirectional: bool = False,
                 pad_idx: int = PAD_IDX,
                 emb_matrix=None,          # (vocab_size, emb_dim) 预训练词向量
                 freeze_emb: bool = False,  # True：预训练向量不参与更新
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT,
                 mlp_hidden: int = MLP_HIDDEN,
                 pooling: str = POOLING):
        super().__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if emb_matrix is not None:
            # 方式 B：把 Word2Vec 预训练矩阵拷进来当初始值（默认继续微调）
            self.embedding.weight.data.copy_(torch.as_tensor(emb_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = not freeze_emb

        self.rnn = _make_rnn(rnn_type, emb_dim, hidden_dim, bidirectional, num_layers, dropout)

        # 双向时特征维度翻倍：正向隐状态和反向隐状态拼接在一起
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        # MLP 多层感知机分类头
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, input_ids, lengths):
        # 1) 文本向量化：N x L  ->  N x L x d
        emb = self.embedding(input_ids)

        # 2) 变长序列打包：让循环层跳过 <PAD>，避免填补标记污染隐状态
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.rnn(packed)

        # 3) 特征提取：取最后一层的最终隐状态（双向则拼接正/反向）
        #    LSTM 的返回值为 (h_n, c_n) 二元组，这里统一取出 h_n
        h_state = h_n[0] if isinstance(h_n, tuple) else h_n

        if self.pooling == "mean":
            out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                         total_length=input_ids.size(1))
            mask = (torch.arange(input_ids.size(1), device=input_ids.device)[None, :]
                    < lengths.to(input_ids.device)[:, None]).unsqueeze(-1)
            feat = (out * mask).sum(dim=1) / lengths.to(input_ids.device)[:, None].clamp(min=1)
        else:
            if self.bidirectional:
                fwd, bwd = h_state[-2], h_state[-1]
                feat = torch.cat([fwd, bwd], dim=-1)
            else:
                feat = h_state[-1]

        # 4) MLP 分类头
        feat = self.dropout(feat)
        logits = self.mlp(feat)
        return logits


def build_model(rnn_type, vocab_size, num_classes=3, emb_matrix=None, hidden_dim=HIDDEN_DIM,
                emb_dim=EMB_DIM, freeze_emb=False, bidirectional=None, **kwargs):
    """统一建模型的入口

    rnn_type 支持 rnn / lstm / gru（单向）与 birnn / bilstm / bigru（双向），
    名字以 bi 开头的自动设成双向。除循环层外其余结构完全一致，保证对比公平。
    """
    name = rnn_type.lower()
    bidir = name.startswith("bi") if bidirectional is None else bidirectional
    return RecSentimentClassifier(
        vocab_size=vocab_size, num_classes=num_classes, emb_dim=emb_dim,
        hidden_dim=hidden_dim, rnn_type=name, bidirectional=bidir,
        emb_matrix=emb_matrix, freeze_emb=freeze_emb, **kwargs)


def count_parameters(model):
    """分块统计参数量：嵌入层 / 循环层 / MLP 头

    等参数量对照实验只看循环层那一栏，所以要单独拆出来统计。
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.embedding.weight.numel()
    rnn = sum(p.numel() for n, p in model.named_parameters() if n.startswith("rnn"))
    return {"total": total, "embedding": emb, "rnn": rnn, "mlp": total - emb - rnn}
