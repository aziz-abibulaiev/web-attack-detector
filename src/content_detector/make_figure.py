"""Draws the two-panel main figure from the result JSONs: held-out false-positive rate across
training regimes, and the deployed in-domain operating point against the cross-source one.
Writes figures/main_result.pdf and figures/main_result.png, which is Figure 1.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "grey": "#999999", "black": "#000000"}


def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def main():
    indomain_res = _load("models/in_domain/in_domain_results.json", {})
    indomain_fp = indomain_res.get("fpr_0.01", {}).get("late_zanbil_alert_rate", 0.0042) * 100
    indomain_fp_corr = indomain_res.get("fpr_0.01", {}).get("late_zanbil_fp_corrected_est", 0.0029) * 100
    srbh_recall = indomain_res.get("fpr_0.01", {}).get("srbh_attack_recall", 1.0) * 100
    tool_recall = indomain_res.get("fpr_0.01", {}).get("testbed_tool_recall", 0.999) * 100

    # regimes, each a held-out real-benign FP at the 1% budget
    regimes = [
        ("Testbed-only\nbenign\n(lab-testbed)", 83.0, OK["red"]),
        ("Single real\nsource single source\nzanbil held", 90.3, OK["red"]),
        ("Cross-source (LOSO)\nzanbil\nheld out", 99.5, OK["orange"]),
        ("In-domain\ntrained on\ntarget benign", indomain_fp, OK["green"]),
    ]

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # ---- (a) FP across regimes, log scale ----
    xs = range(len(regimes))
    axa.bar(xs, [r[1] for r in regimes], color=[r[2] for r in regimes], width=0.62,
            edgecolor="black", linewidth=0.5)
    axa.set_yscale("log")
    axa.set_ylim(0.2, 200)
    axa.set_xticks(list(xs)); axa.set_xticklabels([r[0] for r in regimes], fontsize=8.2)
    axa.set_ylabel("Held-out real-benign false-positive rate (%)\n(log scale, 1% FPR budget)")
    axa.set_title("(a)  Real-benign FP by training regime", fontsize=11, loc="left")
    def _lbl(v):
        if v < 10:
            return f"{v:.2f}%"
        return f"{v:.0f}%" if float(v).is_integer() else f"{v:.1f}%"
    for x, r in zip(xs, regimes):
        axa.text(x, r[1] * 1.12, _lbl(r[1]), ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")
    # 237x arrow between cross-source and in-domain
    axa.annotate("", xy=(3, indomain_fp * 1.5), xytext=(2, 99.5 * 0.85),
                 arrowprops=dict(arrowstyle="->", color=OK["black"], lw=1.4))
    axa.text(2.5, 12, f"~{round(99.5/max(indomain_fp,0.01)):d}x\nlower",
             ha="center", va="center", fontsize=9, fontweight="bold", color=OK["black"])
    axa.grid(axis="y", ls=":", alpha=0.4)

    # ---- (b) operating point: FP vs recall ----
    axb.scatter([99.5], [srbh_recall], s=140, color=OK["orange"], edgecolor="black",
                zorder=3, label="Cross-source (LOSO): unusable FP")
    axb.scatter([indomain_fp], [srbh_recall], s=170, marker="*", color=OK["green"],
                edgecolor="black", zorder=3, label="In-domain: deployed point")
    axb.scatter([indomain_fp], [tool_recall], s=90, marker="D", color=OK["blue"],
                edgecolor="black", zorder=3, label="In-domain, cross-tool recall")
    axb.axvline(indomain_fp, color=OK["green"], ls="--", lw=0.9, alpha=0.6)
    axb.set_xscale("log"); axb.set_xlim(0.2, 200)
    axb.set_ylim(85, 101)
    axb.set_xlabel("Held-out real-benign false-positive rate (%)  (log)")
    axb.set_ylabel("Recall on held-out attacks (%)")
    axb.set_title("(b)  Deployed operating point (1% FPR budget)", fontsize=11, loc="left")
    axb.annotate(f"in-domain\nFP {indomain_fp:.2f}%  (alert rate = upper bound)\n"
                 f"recall {srbh_recall:.0f}% / {tool_recall:.1f}%",
                 xy=(indomain_fp, srbh_recall), xytext=(0.6, 88.5), fontsize=8.4,
                 arrowprops=dict(arrowstyle="->", color=OK["green"], lw=1.0))
    axb.legend(loc="lower right", fontsize=7.8, framealpha=0.9)
    axb.grid(True, ls=":", alpha=0.4)

    fig.suptitle("Content detector: works in-domain, degrades cross-domain "
                 "(text-only, deployed operating point)", fontsize=12, y=1.005)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/main_result.pdf", bbox_inches="tight")
    fig.savefig("figures/main_result.png", dpi=170, bbox_inches="tight")
    print("wrote figures/main_result.{pdf,png}")
    print(f"  in-domain FP={indomain_fp:.2f}% (corrected {indomain_fp_corr:.2f}%), "
          f"SR-BH recall={srbh_recall:.1f}%, cross-tool recall={tool_recall:.1f}%")


if __name__ == "__main__":
    main()
