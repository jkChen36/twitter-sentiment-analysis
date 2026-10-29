"""在不重新训练的前提下，依据 outputs/metrics.json 重新生成图表与报告"""
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config as C
    from src.report import build_report
    from src.run_experiments import build_figures
else:
    from . import config as C
    from .report import build_report
    from .run_experiments import build_figures


def main():
    path = os.path.join(C.OUTPUTS, "metrics.json")
    if not os.path.exists(path):
        print("未找到 metrics.json，请先运行 python -m src.run_experiments")
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    main = payload.get("main", {})
    fair = payload.get("fair", {})
    w2v = payload.get("w2v", {})

    figs = {}
    len_fig = os.path.join(C.FIG_DIR, "01_length_distribution.png")
    if os.path.exists(len_fig):
        figs["length"] = len_fig
    figs.update(build_figures(main, fair, w2v))
    p = build_report(payload, figs)
    print("报告已重新生成：", p)


if __name__ == "__main__":
    main()
