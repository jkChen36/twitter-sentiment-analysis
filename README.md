# Twitter 文本情感分析：单向 vs 双向循环网络

用 Twitter 推文做三分类情感判断（负面 / 中性 / 正面），对比单向循环网络（RNN / LSTM / GRU）和双向循环网络（BiRNN / BiLSTM / BiGRU）的差别。

数据集是 Twitter Entity Sentiment Analysis，74,657 条推文。原始标签有四类是 Positive / Negative / Neutral / Irrelevant，按数据集说明把 Irrelevant 并入 Neutral，统一成三分类。样本按类别分层划分为训练集 59,725 条、验证集 7,466 条、测试集 7,466 条。

## 结果

测试集 7,466 条，指标为 Accuracy / Macro-F1：

| 模型 | 方向 | Accuracy | Macro-F1 | 参数量 |
| --- | --- | --- | --- | --- |
| RNN | 单向 | 0.8575 | 0.8548 | 2,731,139 |
| LSTM | 单向 | 0.8799 | 0.8784 | 2,830,211 |
| GRU | 单向 | 0.8837 | 0.8818 | 2,797,187 |
| BiRNN | 双向 | 0.8749 | 0.8732 | 2,772,355 |
| BiLSTM | 双向 | 0.8899 | 0.8885 | 2,970,499 |
| BiGRU | 双向 | **0.8962** | **0.8948** | 2,904,451 |

双向平均比单向高 1.33 个百分点的准确率。另外做了两组对照：

- 等参数量对照：把单向隐层放大到循环层参数与双向持平，双向在三组里仍然全部领先，说明提升来自信息流方向，不是参数变多。
- 向量化方案对照：Word2Vec 预训练平均 0.8883，比随机初始化的 nn.Embedding（0.8804）高 0.79 个百分点，收敛也更快。

完整的报告（含 9 张图、混淆矩阵和答辩问答页）在 `docs/Twitter文本情感分析实验报告（答辩版）.docx`。

## 跑起来

环境：Python 3.12 + PyTorch 2.14，其余依赖见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

数据集不在仓库里，需要自己下载两个 csv 放到 `data/raw/`：

```text
data/raw/twitter_training.csv
data/raw/twitter_validation.csv
```

原始来源：https://github.com/kaushik2103/Twitter-Sentiment-Analysis
（如果原始地址打不开，搜 Twitter Entity Sentiment Analysis dataset，或找 HuggingFace 上的同名数据集。）

然后：

```bash
# 完整实验：6 个主模型 + 3 个等参数量对照 + 6 个 Word2Vec 对照
# GPU 上大约 15 分钟
python -m src.run_experiments --epochs 12

# 先确认链路能跑通（小样本、2 轮）
python -m src.run_experiments --max-samples 6000 --epochs 2 --skip-fair --skip-w2v

# 用训好的权重对单条文本做推理
python -m src.predict --text "this game is absolutely amazing"
```

常用的参数：`--max-len`（句子统一长度 L，默认 40）、`--emb-dim`（词向量维度 d，默认 128）、`--hidden`（隐层维度）、`--batch-size`。CPU 机器上可以设 `FORCE_CPU=1`。

## 目录说明

```text
src/
  config.py           超参集中在这里：L、d、hidden、划分比例、训练设置
  data_utils.py       清洗分词、建词表、截断补齐、分层划分
  word2vec_utils.py   gensim Word2Vec 预训练（没装 gensim 时自动用内置 SGNS）
  models.py           嵌入层 + 循环层 + MLP 分类头
  train.py            训练循环、交叉熵、早停、指标
  run_experiments.py  实验主入口，跑完全部对比并写 metrics.json
  viz.py / report.py  出图和生成 HTML 报告
  predict.py          单条文本推理
docs/                 实验报告（Word）与结果图
```

关键实现有三处值得看：

1. `pad_or_truncate` —— 长句截断、短句补 `<PAD>`，同时返回真实长度。
2. `RecSentimentClassifier.forward` —— 用 `pack_padded_sequence` 让循环层跳过填补位置；双向时取 `h_n[-2]` 和 `h_n[-1]` 拼接；LSTM 的 `h_n` 是 `(h, c)` 二元组，要先取 `h`。
3. `train_model` —— 按验证集准确率保存最优权重并早停，测试集只在最后评一次。
