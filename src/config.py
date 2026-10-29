"""实验全局配置：Twitter 文本情感分析（单向 RNN  vs  双向 RNN）"""
import os

# ------------------------------ 路径 ------------------------------
# 归档后目录被拆分为"成果类/过程类"结构，允许用环境变量覆盖默认布局；
# 不设置环境变量时行为与原来完全一致（ROOT = 源码目录的上一级）。
ROOT = os.environ.get("TWITTER_SENT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.environ.get("TWITTER_SENT_DATA_RAW") or os.path.join(ROOT, "data", "raw")
DATA_PROC = os.environ.get("TWITTER_SENT_DATA_PROC") or os.path.join(ROOT, "data", "processed")
OUTPUTS = os.environ.get("TWITTER_SENT_OUTPUTS") or os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUTS, "figures")
CKPT_DIR = os.path.join(OUTPUTS, "checkpoints")
for _d in (DATA_PROC, OUTPUTS, FIG_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_RAW, "twitter_training.csv")
VALID_CSV = os.path.join(DATA_RAW, "twitter_validation.csv")

# --------------------------- 数据预处理 ---------------------------
# 预先定义句子最大长度 L：长度 > L 截断(truncate)，长度 < L 补 <PAD>(padding)
MAX_LEN = 40               # L：单句统一长度（全量数据 P90=41，取 40 时截断比例约 10%）
MIN_FREQ = 3               # 词频阈值，低于该值的词映射为 <UNK>
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1

# 标签：原始 4 类（Positive/Negative/Neutral/Irrelevant）
# 实验统一为 3 类情感：Irrelevant 视为 Neutral（与数据集官方说明一致）
LABELS = ["Negative", "Neutral", "Positive"]
LABEL_MAP = {"Negative": 0, "Neutral": 0 + 1, "Irrelevant": 1, "Positive": 2}

# 数据集划分比例：训练集 / 验证集 / 测试集
SPLIT_RATIO = (0.8, 0.1, 0.1)
SEED = 42
MAX_SAMPLES = None         # 调试用：None 表示使用全量数据

# --------------------------- 文本向量化 ---------------------------
EMB_DIM = 128              # d：embedding 维度（2 的整数次幂，可选 128 / 256）
W2V_DIM = 128              # Word2Vec 词向量维度
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 5
W2V_NEGATIVE = 5

# ----------------------------- 模型 -----------------------------
HIDDEN_DIM = 128           # 循环层隐状态维度
NUM_LAYERS = 1
DROPOUT = 0.5
MLP_HIDDEN = 64            # MLP 分类头隐层宽度
POOLING = "last"           # last：取最后时刻隐状态；mean：按 mask 做均值池化

# ----------------------------- 训练 -----------------------------
BATCH_SIZE = 256
NUM_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
PATIENCE = 3               # 验证集 early stopping 耐心值
DEVICE = "cuda" if os.environ.get("FORCE_CPU") is None and __import__("torch").cuda.is_available() else "cpu"

RNN_TYPES = ["rnn", "lstm", "gru"]   # 单向特征提取结构
BI_TYPES = ["birnn", "bilstm", "bigru"]  # 双向特征提取结构
