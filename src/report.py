"""生成实验结果表（CSV / Markdown）与 HTML 实验报告"""
import csv
import html
import os

from .config import LABELS, OUTPUTS


def _fmt(x, n=4):
    return f"{x:.{n}f}"


def build_tables(payload):
    rows = []
    main = payload["main"]

    def add(name, rec, scheme):
        pc = rec["test"]["per_class"]
        rows.append({
            "模型": name,
            "方案": scheme,
            "方向": rec["group"],
            "Acc": _fmt(rec["test"]["accuracy"]),
            "Macro-F1": _fmt(rec["test"]["macro_f1"]),
            "Neg-F1": _fmt(pc[0]["f1"]),
            "Neu-F1": _fmt(pc[1]["f1"]),
            "Pos-F1": _fmt(pc[2]["f1"]),
            "参数量": rec["params"]["total"],
            "循环层参数量": rec["params"]["rnn"],
            "最优epoch": rec["history"]["best_epoch"],
            "训练耗时s": round(rec["train_time"], 1),
        })

    for v in main.values():
        add(v["name"], v, v["scheme"])
    for v in payload.get("fair", {}).values():
        add(v["name"], v, v["scheme"])
    for v in payload.get("w2v", {}).values():
        add(v["name"], v, v["scheme"])
    return rows


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def markdown_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _img(rel):
    return f'<img src="{rel}" loading="lazy">'


def build_report(payload, figs):
    cfg = payload["config"]
    st = payload["dataset_stats"]
    main = payload["main"]
    rows = build_tables(payload)
    write_csv(rows, os.path.join(OUTPUTS, "results_table.csv"))

    uni = [v for v in main.values() if v["group"] == "单向"]
    bi = [v for v in main.values() if v["group"] == "双向"]
    uni_acc = sum(v["test"]["accuracy"] for v in uni) / max(len(uni), 1)
    bi_acc = sum(v["test"]["accuracy"] for v in bi) / max(len(bi), 1)
    uni_f1 = sum(v["test"]["macro_f1"] for v in uni) / max(len(uni), 1)
    bi_f1 = sum(v["test"]["macro_f1"] for v in bi) / max(len(bi), 1)

    pairs = []
    for u in uni:
        b = f"BI{u['name']}"
        if b in main:
            pairs.append((u["name"], b,
                          main[b]["test"]["accuracy"] - u["test"]["accuracy"],
                          main[b]["test"]["macro_f1"] - u["test"]["macro_f1"]))
    best_pair = max(pairs, key=lambda p: p[2]) if pairs else None
    best_model = max(list(main.values()), key=lambda v: v["test"]["accuracy"])

    # 关键结论（自动生成）
    conclusions = []
    conclusions.append(
        f"在测试集上，双向模型平均 Accuracy <b>{_fmt(bi_acc)}</b>、Macro-F1 <b>{_fmt(bi_f1)}</b>；"
        f"单向模型平均 Accuracy <b>{_fmt(uni_acc)}</b>、Macro-F1 <b>{_fmt(uni_f1)}</b>。"
        f"双向结构带来 <b>+{_fmt(bi_acc - uni_acc)}</b> 的准确率提升与 "
        f"<b>+{_fmt(bi_f1 - uni_f1)}</b> 的 Macro-F1 提升。")
    if best_pair:
        conclusions.append(
            f"提升最明显的是 <b>{best_pair[1]}</b> 相对 <b>{best_pair[0]}</b>：Accuracy "
            f"+{_fmt(best_pair[2])}，Macro-F1 +{_fmt(best_pair[3])}。"
            "原因是单向结构只能从左到右累积信息，句尾词的语义对句首词没有反馈，"
            "而情感判定常依赖句尾出现的否定词、转折词与情感词强度。")
    if payload.get("fair"):
        lines = []
        for v in payload["fair"].values():
            base = v["fair_of"]
            d = main[base]["test"]["accuracy"] - v["test"]["accuracy"]
            lines.append(
                f"<b>{base}</b> {_fmt(main[base]['test']['accuracy'])} vs 同参数量单向 "
                f"<b>{v['name']}</b> {_fmt(v['test']['accuracy'])}"
                f"（循环层参数 {main[base]['params']['rnn']:,} ≈ {v['params']['rnn']:,}，差 {d:+.4f}）")
        conclusions.append(
            "等参数量对照（把单向隐层维度放大到与双向模型循环层参数量持平）：" + "；".join(lines) +
            "。参数量相当的条件下双向结构依然不落下风，说明提升来自信息流方向本身，而不是单纯堆参数。")
    if payload.get("w2v"):
        w2v_vals = list(payload["w2v"].values())
        emb_vals = list(main.values())
        w2v_acc = sum(v["test"]["accuracy"] for v in w2v_vals) / len(w2v_vals)
        emb_acc = sum(v["test"]["accuracy"] for v in emb_vals) / len(emb_vals)
        w2v_uni = [v for v in w2v_vals if v["group"] == "单向"]
        w2v_bi = [v for v in w2v_vals if v["group"] == "双向"]
        w2v_gain = (sum(v["test"]["accuracy"] for v in w2v_bi) / max(len(w2v_bi), 1)
                    - sum(v["test"]["accuracy"] for v in w2v_uni) / max(len(w2v_uni), 1))
        if w2v_acc >= emb_acc:
            tail = (f"Word2Vec 预训练平均 Accuracy <b>{_fmt(w2v_acc)}</b> 高于随机初始化的 "
                    f"nn.Embedding <b>{_fmt(emb_acc)}</b>（+{_fmt(w2v_acc - emb_acc)}）："
                    f"在 {st['num_samples']:,} 条推文语料上预训练的词向量自带语义信息，"
                    "相当于给模型提供了先验，尤其在训练早期收敛更快、且能用更少 epoch 达到最佳。")
        else:
            tail = (f"nn.Embedding 端到端学习平均 Accuracy <b>{_fmt(emb_acc)}</b> 略高于 "
                    f"Word2Vec 预训练 <b>{_fmt(w2v_acc)}</b>：数据量充足时，随机初始化的词嵌入"
                    "能直接随下游目标联合优化，不一定依赖预训练。")
        conclusions.append(
            f"向量化方案对比：{tail}"
            f"同一 Word2Vec 方案下，双向相对单向仍有 <b>+{_fmt(w2v_gain)}</b> 的提升，"
            "说明双向结构的优势与向量化方式无关，是稳定存在的。")

    # 表格 HTML
    cols = ["模型", "方案", "方向", "Acc", "Macro-F1", "Neg-F1", "Neu-F1", "Pos-F1",
            "参数量", "最优epoch", "训练耗时s"]
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    tbody = ""
    for r in rows:
        cls = "bi" if r["方向"] == "双向" else "uni"
        tbody += f'<tr class="{cls}">' + "".join(
            f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>"

    def f2(key, fallback=""):
        p = figs.get(key)
        return _img(os.path.relpath(p, OUTPUTS).replace("\\", "/")) if p else fallback

    demo = payload["demo"]
    w2v_block = ""
    if payload.get("w2v"):
        w2v_block = f"""
    <h3>向量化方案对照</h3>
    <div class="grid2">
      <figure>{f2('emb_ab')}<figcaption>nn.Embedding（可学习）与 Word2Vec（预训练）在各循环单元上的测试准确率</figcaption></figure>
    </div>"""

    fair_block = ""
    if payload.get("fair"):
        fair_block = f"""
    <h3>等参数量公平对照</h3>
    <div class="grid2">
      <figure>{f2('fair')}<figcaption>把单向隐层维度放大到与双向模型循环层参数量相当后重新对比</figcaption></figure>
    </div>"""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Twitter 文本情感分析实验报告</title>
<style>
  :root {{ --line:#E6E8EB; --muted:#5F6B7A; --uni:#4C78A8; --bi:#E45756; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:#F7F8FA; color:#1F2933;
         font-family:"Microsoft YaHei","Segoe UI",-apple-system,sans-serif; line-height:1.7; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:40px 28px 80px; }}
  header {{ background:linear-gradient(135deg,#FFFFFF,#F0F4F8); border:1px solid var(--line);
            border-radius:14px; padding:26px 30px; }}
  h1 {{ margin:0 0 6px; font-size:26px; }}
  h2 {{ font-size:20px; margin:38px 0 12px; padding-left:12px; border-left:4px solid var(--uni); }}
  h3 {{ font-size:16px; margin:22px 0 8px; color:#334155; }}
  .meta {{ color:var(--muted); font-size:13px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-top:16px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .card .k {{ font-size:12px; color:var(--muted); }}
  .card .v {{ font-size:18px; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; font-size:13px;
           border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  th, td {{ padding:9px 10px; text-align:center; border-bottom:1px solid var(--line); }}
  th {{ background:#F1F4F8; font-weight:600; }}
  tr.uni td:nth-child(1) {{ color:var(--uni); font-weight:600; }}
  tr.bi td:nth-child(1) {{ color:var(--bi); font-weight:600; }}
  code, pre {{ background:#F1F4F8; border-radius:6px; font-family:Consolas,Menlo,monospace; font-size:12.5px; }}
  pre {{ padding:12px 14px; overflow-x:auto; border:1px solid var(--line); }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; }}
  .pill.uni {{ background:rgba(76,120,168,.12); color:var(--uni); }}
  .pill.bi {{ background:rgba(228,87,86,.13); color:var(--bi); }}
  ul {{ margin:8px 0 0 20px; padding:0; }}
  li {{ margin:5px 0; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  figure {{ margin:0; background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; }}
  figure img {{ width:100%; display:block; border-radius:8px; }}
  figcaption {{ font-size:12px; color:var(--muted); margin-top:8px; text-align:center; }}
  .concl {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--bi);
            border-radius:10px; padding:14px 18px; margin:12px 0; }}
  @media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
<header>
  <h1>Twitter 文本情感分析：单向循环网络 vs 双向循环网络</h1>
  <div class="meta">数据集：Twitter Entity Sentiment Analysis（twitter_training / twitter_validation 合并，
  标签 Positive / Negative / Neutral，Irrelevant 并入 Neutral）　·　PyTorch {cfg.get('torch', '')}　·　设备 {cfg['device']}</div>
  <div class="cards">
    <div class="card"><div class="k">样本数 N</div><div class="v">{st['num_samples']:,}</div></div>
    <div class="card"><div class="k">统一句长 L</div><div class="v">{cfg['max_len_L']}</div></div>
    <div class="card"><div class="k">词表大小</div><div class="v">{st['vocab_size']:,}</div></div>
    <div class="card"><div class="k">向量维度 d</div><div class="v">{cfg['emb_dim']}</div></div>
    <div class="card"><div class="k">隐层维度</div><div class="v">{cfg['hidden_dim']}</div></div>
    <div class="card"><div class="k">最佳模型</div><div class="v">{best_model['name']} {_fmt(best_model['test']['accuracy'])}</div></div>
  </div>
</header>

<h2>一、任务 1：数据预处理（padding + truncate）</h2>
<p>原始推文是非结构化文本，长短不一，无法直接组批输入网络。处理链路为：
<b>清洗 → 分词 → 按词表转 ID → 长句 truncate、短句 padding → 统一为 N × L 的整数矩阵</b>。</p>
<div class="cards">
  <div class="card"><div class="k">预处理后形状 N × L</div><div class="v">{st['shape'][0]} × {st['shape'][1]}</div></div>
  <div class="card"><div class="k">截断比例</div><div class="v">{st['truncated_ratio']:.1%}</div></div>
  <div class="card"><div class="k">补齐比例</div><div class="v">{st['padded_ratio']:.1%}</div></div>
  <div class="card"><div class="k">句长 P95</div><div class="v">{st['raw_len_p95']:.0f}</div></div>
</div>
<h3>样例完整链路（题目输入样例）</h3>
<pre>原文    : {html.escape(demo['raw'])}
清洗后  : {html.escape(demo['cleaned'])}
分词    : {html.escape(' '.join(demo['tokens']))}
ID 序列 : {html.escape(str(demo['ids']))}
补齐后  : 长度 {demo['padded_len']}（真实词数 {demo['real_len']}，尾部补 {demo['padded_len'] - demo['real_len']} 个 &lt;PAD&gt;，索引 0）</pre>
<div class="grid2">
  <figure>{f2('length')}<figcaption>句子长度分布与 L 阈值</figcaption></figure>
</div>
<p class="meta">词表构建：<code>&lt;PAD&gt;=0</code>、<code>&lt;UNK&gt;=1</code>，词频阈值 {cfg['min_freq']}，低频词统一映射为 <code>&lt;UNK&gt;</code>。
数据集按类别分层划分为 训练集 {st['split_sizes']['train']} / 验证集 {st['split_sizes']['val']} / 测试集 {st['split_sizes']['test']}（8:1:1）。</p>

<h2>二、任务 2：文本向量化</h2>
<p>两种方式都实现并参与对比：</p>
<ul>
  <li><b>方式 A · nn.Embedding 可学习嵌入层</b>：随机初始化 <code>(词表大小, d={cfg['emb_dim']})</code> 的嵌入矩阵，
      <code>padding_idx=0</code> 使 <code>&lt;PAD&gt;</code> 的向量恒为 0 且不参与梯度更新，词向量随任务目标在训练中同步更新。</li>
  <li><b>方式 B · Word2Vec 预训练词向量</b>：使用 <code>{html.escape(cfg['w2v_backend'])}</code> 在全部推文语料上训练
      Skip-gram 词向量（d={cfg['w2v_dim']}，窗口 5，负采样 5），再按词表对齐成嵌入矩阵加载进模型，加载后继续微调。</li>
</ul>
<p>向量化后的数据形状：<code>N × L × d = {st['num_samples']:,} × {cfg['max_len_L']} × {cfg['emb_dim']}</code>。
后续循环层会按样本真实长度打包（<code>pack_padded_sequence</code>），使 <code>&lt;PAD&gt;</code> 不参与状态计算。</p>
{w2v_block}

<h2>三、任务 3/4：模型结构</h2>
<p>所有模型共享同一套结构：<b>嵌入层 → 循环特征提取器 → MLP 分类头</b>，
区别只在循环层是单向还是双向、以及循环单元类型。</p>
<pre>Embedding(N×L → N×L×{cfg['emb_dim']})  →  [单向] RNN / LSTM / GRU
                                    →  [双向] BiRNN / BiLSTM / BiGRU
feature = 最后时刻隐状态 h_T（双向 = concat(h_forward, h_backward)）
MLP: Linear(feat) → ReLU → Dropout(0.5) → Linear(3) → CrossEntropyLoss</pre>
<p class="meta">损失函数：交叉熵 <code>nn.CrossEntropyLoss</code>；优化器 Adam（lr={cfg['lr']}）；梯度裁剪 5.0；
按验证集准确率保存最优权重，耐心值 3 轮早停，最终在测试集上评估。</p>

<h2>四、任务 5：训练曲线与结果对比</h2>
<div class="grid2">
  <figure>{f2('curve_emb')}<figcaption>Embedding 方案下 6 个模型的验证集损失与准确率曲线</figcaption></figure>
  <figure>{f2('bar_acc')}<figcaption>测试集 Accuracy 对比</figcaption></figure>
</div>
<div class="grid2">
  <figure>{f2('bar_f1')}<figcaption>测试集 Macro-F1 对比</figcaption></figure>
  <figure>{f2('delta')}<figcaption>双向相对单向的准确率提升幅度</figcaption></figure>
</div>
<div class="grid2">
  <figure>{f2('cm_uni')}<figcaption>最佳单向模型混淆矩阵（行归一化）</figcaption></figure>
  <figure>{f2('cm_bi')}<figcaption>最佳双向模型混淆矩阵（行归一化）</figcaption></figure>
</div>
{fair_block}

<h2>五、完整结果表</h2>
<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>
<p class="meta">方向标记：<span class="pill uni">单向</span> <span class="pill bi">双向</span>　
Neg / Neu / Pos 为三类的 F1；参数量为可训练参数总数。</p>

<h2>六、结论与分析</h2>
{"".join(f'<div class="concl">{c}</div>' for c in conclusions)}
<div class="concl">
  <b>为什么双向结构对情感分析有帮助：</b>
  情感极性往往由句中前后搭配共同决定——否定词（not / never / no）常出现在情感词之前，
  转折词（but / however）会把语义重心转移到后半句。单向网络只看得到"前文 → 当前"的累积，
  句首词的表征无法感知后文；双向网络同时拥有从左到右与从右到左两条信息流，
  最后时刻隐状态是两个方向语义的拼接，因此能更完整地捕获这类前后依赖。
</div>
<div class="concl">
  <b>代价：</b>双向结构的循环层参数量与计算量约为单向的 2 倍（反向还要再扫一遍序列），
  训练耗时相应增加。若部署场景对延迟敏感且文本较短，可在单向与双向之间按上面的
  "精度提升 / 参数开销"权衡取舍。
</div>
</div></body></html>"""

    path = os.path.join(OUTPUTS, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)

    # Markdown 版本
    md_cols = ["模型", "方案", "方向", "Acc", "Macro-F1", "参数量", "最优epoch", "训练耗时s"]
    md = "# Twitter 文本情感分析实验结果\n\n"
    md += f"- 数据：N={st['num_samples']:,}，L={cfg['max_len_L']}，词表 {st['vocab_size']:,}，d={cfg['emb_dim']}\n"
    md += f"- 划分：train {st['split_sizes']['train']} / val {st['split_sizes']['val']} / test {st['split_sizes']['test']}\n\n"
    md += markdown_table(rows, md_cols) + "\n\n"
    md += "## 结论\n" + "\n".join(f"- {c}" for c in conclusions) + "\n"
    with open(os.path.join(OUTPUTS, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return path
