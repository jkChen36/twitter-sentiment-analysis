"""实验主入口：一次性跑通任务 1~5 的全部对比实验"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config as C
    from src import viz
    from src.data_utils import (build_dataset, describe_dataset, encode_all,
                                pad_or_truncate, tokenize, clean_text)
    from src.models import build_model, count_parameters
    from src.train import make_loaders, train_model
    from src.word2vec_utils import backend_name, train_word2vec
    from src.report import build_report
else:
    from . import config as C
    from . import viz
    from .data_utils import (build_dataset, describe_dataset, encode_all,
                             pad_or_truncate, tokenize, clean_text)
    from .models import build_model, count_parameters
    from .train import make_loaders, train_model
    from .word2vec_utils import backend_name, train_word2vec
    from .report import build_report

DEMO_TEXT = ("one vote can make all the difference Anil Kapoor answers Modi's "
             "election 2019 clarion call extends")


def set_seed(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_matched_hidden(rnn_type, emb_dim, target_rnn_params, vocab_size, max_h=512):
    """为单向模型搜索隐层维度 h，使其循环层参数量最接近给定的双向模型参数量

    等参数量对照用的就是它：双向天生比单向多一倍循环层参数，
    这里把单向的 h 加大（实验里 RNN / LSTM / GRU 都落在 200 附近），
    把两者的循环层参数拉平，再比较准确率。
    """
    best_h, best_gap = None, float("inf")
    for h in range(16, max_h + 1, 4):
        m = build_model(rnn_type, vocab_size, emb_dim=emb_dim, hidden_dim=h, bidirectional=False)
        p = count_parameters(m)["rnn"]
        gap = abs(p - target_rnn_params)
        if gap < best_gap:
            best_gap, best_h = gap, h
        if p > target_rnn_params * 1.02:
            break
    return best_h, best_gap


def run_group(name, rnn_types, dataset, loaders, emb_matrix, emb_dim, hidden_dim,
              epochs, freeze=False, verbose=True, scheme="nn.Embedding"):
    """训练一组模型，返回结果记录

    每组开跑前都固定随机种子，六个模型除了循环层之外结构完全一致，
    这样对比出来的差异只来自循环单元类型和信息流方向。
    """
    vocab_size = len(dataset["vocab"])
    group = {}
    for rt in rnn_types:
        tag = f"{scheme.split('.')[-1]}/{rt.upper()}"
        set_seed(C.SEED)
        model = build_model(rt, vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                            emb_matrix=emb_matrix, freeze_emb=freeze)
        params = count_parameters(model)
        t0 = time.time()
        hist, test_m, model = train_model(model, loaders, epochs=epochs, verbose=verbose, tag=tag)
        ckpt = os.path.join(C.CKPT_DIR, f"{scheme.replace('.', '_')}_{rt.lower()}.pt")
        torch.save({"state_dict": model.state_dict(),
                    "test": {k: v for k, v in test_m.items() if k != "confusion_matrix"},
                    "params": params,
                    "arch": {"rnn_type": rt, "emb_dim": emb_dim, "hidden_dim": hidden_dim,
                             "freeze_emb": freeze, "scheme": scheme}}, ckpt)
        rec = {
            "name": rt.upper(),
            "scheme": scheme,
            "group": "双向" if rt.lower().startswith("bi") else "单向",
            "history": hist,
            "test": test_m,
            "params": params,
            "hidden_dim": hidden_dim,
            "emb_dim": emb_dim,
            "freeze_emb": freeze if emb_matrix is not None else False,
            "train_time": time.time() - t0,
            "ckpt": ckpt,
        }
        group[rt.upper()] = rec
        print(f"  ✓ {tag:<22} test_acc={test_m['accuracy']:.4f} "
              f"macro_f1={test_m['macro_f1']:.4f} params={params['total']:,} "
              f"({rec['train_time']:.0f}s)", flush=True)
    return group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=C.MAX_SAMPLES)
    ap.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    ap.add_argument("--emb-dim", type=int, default=C.EMB_DIM)
    ap.add_argument("--hidden", type=int, default=C.HIDDEN_DIM)
    ap.add_argument("--max-len", type=int, default=C.MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--skip-w2v", action="store_true")
    ap.add_argument("--skip-fair", action="store_true")
    args = ap.parse_args()

    if args.max_samples:
        C.MAX_SAMPLES = args.max_samples
    C.EMB_DIM = C.W2V_DIM = args.emb_dim
    C.MAX_LEN = args.max_len
    C.BATCH_SIZE = args.batch_size

    print("=" * 78)
    print("实验：Twitter 文本情感分析 —— 单向循环网络 vs 双向循环网络")
    print("=" * 78)
    print(f"device = {C.DEVICE} | emb_dim(d) = {C.EMB_DIM} | hidden = {args.hidden} "
          f"| L = {C.MAX_LEN} | epochs = {args.epochs} | batch = {C.BATCH_SIZE}\n")

    # ---------------- 任务 1：数据预处理 ----------------
    print("[任务 1] 数据预处理：清洗 -> 分词 -> truncate + padding 统一长度 L")
    t0 = time.time()
    dataset = build_dataset(max_len=C.MAX_LEN, min_freq=C.MIN_FREQ)
    dataset["vocab"].save(os.path.join(C.OUTPUTS, "vocab.json"))
    stats = describe_dataset(dataset)
    print(f"  样本数 N = {stats['num_samples']:,}，统一后数据形状 N × L = {stats['shape']}")
    print(f"  词表大小 = {stats['vocab_size']:,}；原始句长 均值 {stats['raw_len_mean']:.1f} / "
          f"P90 {stats['raw_len_p90']:.0f} / P95 {stats['raw_len_p95']:.0f} / 最长 {stats['raw_len_max']}")
    print(f"  被截断(truncate) 的句子占比 {stats['truncated_ratio']:.2%}，"
          f"被补齐(padding) 的句子占比 {stats['padded_ratio']:.2%}")
    print(f"  类别分布：{stats['class_counts']}")
    print(f"  划分：训练集 {stats['split_sizes']['train']} / 验证集 {stats['split_sizes']['val']} "
          f"/ 测试集 {stats['split_sizes']['test']}  ({time.time() - t0:.1f}s)")

    # 样例演示：题目输入样例的完整预处理链路
    cl = clean_text(DEMO_TEXT)
    toks = tokenize(cl)
    ids = dataset["vocab"].encode(toks)
    padded, real_len = pad_or_truncate(ids, C.MAX_LEN)
    print(f"\n  样例演示：{DEMO_TEXT[:60]}...")
    print(f"    清洗后 -> {cl}")
    print(f"    分词   -> {toks}")
    print(f"    ID序列 -> {ids}")
    print(f"    pad/truncate 后长度 = {len(padded)}（真实词数 {real_len}，"
          f"补齐 {C.MAX_LEN - len(ids) if len(ids) < C.MAX_LEN else 0} 个 <PAD>）")

    fig_len = viz.plot_length_distribution(dataset["raw_token_lengths"], C.MAX_LEN)

    # ---------------- 任务 2：文本向量化 ----------------
    print("\n[任务 2] 文本向量化")
    print(f"  方式 A：nn.Embedding 可学习嵌入层（随训练同步更新），d = {C.EMB_DIM}")
    w2v_matrix = None
    w2v_backend = "未启用"
    if not args.skip_w2v:
        t0 = time.time()
        w2v_matrix, w2v_backend = train_word2vec(dataset["token_lists"], dataset["vocab"],
                                                 dim=C.W2V_DIM)
        print(f"  方式 B：{w2v_backend} 预训练词向量，形状 = {w2v_matrix.shape}，d = {C.W2V_DIM} "
              f"({time.time() - t0:.1f}s)")
        # 词向量相似度抽查（高频 hub 词会与所有词相似，展示时排除前 60 个高频词）
        vocab = dataset["vocab"]
        freq = np.zeros(len(vocab), dtype=np.float64)
        for toks in dataset["token_lists"]:
            for t in toks:
                freq[vocab.stoi.get(t, 1)] += 1
        hub = set(np.argsort(-freq)[:60].tolist())
        probes = ["good", "bad", "love", "hate", "great", "sad"]
        shown = 0
        for probe in probes:
            if probe in vocab.stoi and shown < 4:
                v = w2v_matrix[vocab.stoi[probe]]
                sims = w2v_matrix @ v
                sims[list(hub)] = -np.inf      # 排除高频通用词
                sims[vocab.stoi[probe]] = -np.inf
                sims[0] = sims[1] = -np.inf    # 排除 <PAD>/<UNK>
                near = [vocab.itos[i] for i in np.argsort(-sims)[:5]]
                print(f"    与 '{probe}' 最相似的词：{near}")
                shown += 1

    print(f"  向量化后数据形状：N × L × d = "
          f"{stats['shape'][0]} × {stats['shape'][1]} × {C.EMB_DIM}")

    loaders = make_loaders(dataset["split"], batch_size=C.BATCH_SIZE)
    all_results = {}

    # ---------------- 任务 3~5：单向 vs 双向（Embedding 方案）----------------
    print("\n[任务 3/4/5] 训练：单向 RNN/LSTM/GRU  vs  双向 BiRNN/BiLSTM/BiGRU")
    print("-" * 78)
    main_group = run_group("embedding", C.RNN_TYPES + C.BI_TYPES, dataset, loaders,
                           None, C.EMB_DIM, args.hidden, args.epochs)
    all_results.update(main_group)

    # ---------------- 等参数量公平对照 ----------------
    fair_results = {}
    if not args.skip_fair:
        print("\n[附加] 等参数量公平对照：单向隐层维度放大到与双向模型参数量相当")
        print("-" * 78)
        for base in ["BIRNN", "BILSTM", "BIGRU"]:
            if base not in main_group:
                continue
            target = main_group[base]["params"]["rnn"]
            uni_type = base[2:]
            h, gap = find_matched_hidden(uni_type, C.EMB_DIM, target,
                                         len(dataset["vocab"]))
            g = run_group("fair", [uni_type.lower()], dataset, loaders, None,
                          C.EMB_DIM, h, args.epochs)
            for k, v in g.items():
                v["name"] = f"{k}(h={h})"
                v["fair_of"] = base
                fair_results[f"{k}-h{h}"] = v
        all_results.update(fair_results)

    # ---------------- Word2Vec 方案对照 ----------------
    w2v_results = {}
    if w2v_matrix is not None:
        print("\n[向量化对照] Word2Vec 预训练词向量 + 单向/双向模型")
        print("-" * 78)
        g = run_group("w2v", C.RNN_TYPES + C.BI_TYPES, dataset, loaders, w2v_matrix,
                      C.W2V_DIM, args.hidden, args.epochs, scheme="Word2Vec")
        for k, v in g.items():
            w2v_results[f"{k}@w2v"] = v
        all_results.update(w2v_results)

    # ---------------- 汇总与产出 ----------------
    print("\n[产出] 汇总结果、绘制图表、生成报告")
    payload = {
        "config": {
            "max_len_L": C.MAX_LEN, "emb_dim": C.EMB_DIM, "hidden_dim": args.hidden,
            "batch_size": C.BATCH_SIZE, "epochs": args.epochs, "lr": C.LR,
            "device": C.DEVICE, "min_freq": C.MIN_FREQ, "torch": torch.__version__,
            "split_ratio": list(C.SPLIT_RATIO), "seed": C.SEED,
            "w2v_backend": w2v_backend, "w2v_dim": C.W2V_DIM,
        },
        "dataset_stats": stats,
        "demo": {
            "raw": DEMO_TEXT, "cleaned": cl, "tokens": toks, "ids": ids,
            "padded_len": len(padded), "real_len": real_len,
        },
        "main": main_group,
        "fair": fair_results,
        "w2v": w2v_results,
    }
    with open(os.path.join(C.OUTPUTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)

    figs = {"length": fig_len}
    figs.update(build_figures(main_group, fair_results, w2v_results))

    build_report(payload, figs)
    print("\n完成：报告 outputs/report.html，指标 outputs/metrics.json")


def build_figures(main_group, fair_results, w2v_results):
    figs = {}
    # 训练曲线：按单向/双向分两张
    figs["curve_emb"] = viz.plot_curves(
        {k: v["history"] for k, v in main_group.items()},
        "Embedding 方案：验证集损失 / 准确率曲线（蓝=单向，红=双向）",
        "02_curves_embedding.png")

    def bars(group, metric_key, label, name, extra=None):
        items = [{"name": v["name"] + (extra or ""), "value": v["test"][metric_key],
                  "group": v["group"]} for v in group.values()]
        return viz.plot_group_bars(items, label, label, name, ylabel=label)

    figs["bar_acc"] = bars(main_group, "accuracy", "Test Accuracy",
                           "03_compare_accuracy.png")
    figs["bar_f1"] = bars(main_group, "macro_f1", "Test Macro-F1",
                          "04_compare_macro_f1.png")

    # 提升幅度
    deltas = []
    for uni in ["RNN", "LSTM", "GRU"]:
        bi = f"BI{uni}"
        if uni in main_group and bi in main_group:
            deltas.append((f"{bi} - {uni}",
                           main_group[bi]["test"]["accuracy"] - main_group[uni]["test"]["accuracy"]))
    if deltas:
        figs["delta"] = viz.plot_delta_bars(deltas, "05_delta_accuracy.png")

    # 等参数量对照
    if fair_results:
        items = []
        for k, v in fair_results.items():
            base = v["fair_of"]
            items.append({"name": f"{base}(h={main_group[base]['hidden_dim']})",
                          "value": main_group[base]["test"]["accuracy"], "group": "双向"})
            items.append({"name": v["name"], "value": v["test"]["accuracy"],
                          "group": "单向"})
        figs["fair"] = viz.plot_group_bars(
            items, "acc", "等参数量对照：双向 vs 放大后的单向", "06_fair_param_compare.png",
            ylabel="Test Accuracy")

    # 混淆矩阵：最佳双向 与 最佳单向
    if main_group:
        bi_best = max([v for v in main_group.values() if v["group"] == "双向"],
                      key=lambda v: v["test"]["accuracy"], default=None)
        uni_best = max([v for v in main_group.values() if v["group"] == "单向"],
                       key=lambda v: v["test"]["accuracy"], default=None)
        if bi_best:
            figs["cm_bi"] = viz.plot_confusion(bi_best["test"]["confusion_matrix"],
                                               f"混淆矩阵：{bi_best['name']}（双向最佳）",
                                               "07_confusion_bi.png")
        if uni_best:
            figs["cm_uni"] = viz.plot_confusion(uni_best["test"]["confusion_matrix"],
                                                f"混淆矩阵：{uni_best['name']}（单向最佳）",
                                                "08_confusion_uni.png")

    # 向量化方案对比
    if w2v_results:
        items = []
        for k, v in w2v_results.items():
            items.append({"model": v["name"], "scheme": "Word2Vec", "value": v["test"]["accuracy"]})
        for v in main_group.values():
            items.append({"model": v["name"], "scheme": "nn.Embedding", "value": v["test"]["accuracy"]})
        figs["emb_ab"] = viz.plot_embedding_ab(items, "09_embedding_ab.png")
    return figs


if __name__ == "__main__":
    main()
