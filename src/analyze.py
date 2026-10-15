"""Сводные таблицы, доверительные интервалы по seed, парные сравнения, графики для отчёта ЛР3."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figs"
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
ORDER = ["IQL", "VDN", "QMIX", "IPPO", "MAPPO", "MAA2C", "COMA", "MADDPG", "PPO (1 агент)", "Равные веса"]
METRICS = ["R_mean", "sharpe", "sortino", "cvar5", "max_drawdown", "catalog_acc", "auto_rate", "auto_err",
           "review_rate", "unlabeled", "violations", "sla_breach", "abstain"]


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std() == 0:
        return x.mean(), x.mean(), x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def table(ev, set_name):
    d = ev[(ev.set == set_name) & (ev.tag == "main")]
    rows = []
    for a in ORDER:
        g = d[d.algo == a]
        r = {"algo": a}
        for mtr in METRICS:
            m, lo, hi = ci(g[mtr])
            r[mtr], r[mtr + "_lo"], r[mtr + "_hi"] = m, lo, hi
        r["R_seed_std"] = g.R_mean.std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(RES / "eval_all.csv")
    meta = pd.read_csv(RES / "runs_meta.csv")
    cfg = json.load(open(RES / "config_log.json"))
    out = {}
    for s in ["val", "test", "jamendo_2010_2017", "jamendo_2005_2009"]:
        t = table(ev, s)
        t.to_csv(RES / f"summary_{s}.csv", index=False)
        out[s] = t

    # ---- парное сравнение на тесте с «Равные веса» и с PPO: одинаковые эпизоды (seed оценки 777)
    pair = []
    base = {b: np.mean([np.load(RES / "episode_R" / f"{b}_{s}.npy") for s in range(5)], 0) for b in ["Равные веса", "PPO (1 агент)"]}
    for a in ORDER[:-1]:
        seeds_R = [np.load(RES / "episode_R" / f"{a}_{s}.npy") for s in range(5)]
        for b in ["Равные веса", "PPO (1 агент)"]:
            if a == b:
                continue
            diff_seed = [r.mean() - base[b].mean() for r in seeds_R]      # разность по seed
            m, lo, hi = ci(diff_seed)
            ep_diff = np.mean(seeds_R, 0) - base[b]                       # парная по эпизодам
            w = stats.wilcoxon(ep_diff) if np.any(ep_diff != 0) else None
            pair.append({"algo": a, "vs": b, "delta_R": m, "ci_lo": lo, "ci_hi": hi,
                         "seeds_better": int(sum(d > 0 for d in diff_seed)),
                         "wilcoxon_p_episodes": w.pvalue if w else 1.0})
    pd.DataFrame(pair).to_csv(RES / "paired_vs_baselines.csv", index=False)

    # ---- стресс-тесты
    st = ev[ev.set.str.startswith("stress") & (ev.tag == "main")].groupby(["algo", "set"]).R_mean.mean().unstack()
    st["base (test)"] = out["test"].set_index("algo").R_mean
    st.loc[ORDER].to_csv(RES / "stress_table.csv")

    # ---- абляции лучшего MARL
    best = cfg["best_marl_by_val"]
    ab = pd.read_csv(RES / "ablations.csv")
    full = ev[(ev.algo == best) & (ev.tag == "main")].assign(tag="полная система")
    ab = pd.concat([full[~full.set.str.startswith("stress")], ab])
    abl = ab.groupby(["tag", "set"]).agg(R=("R_mean", "mean"), R_std=("R_mean", "std"), acc=("catalog_acc", "mean"),
                                         review=("review_rate", "mean"), auto_err=("auto_err", "mean"),
                                         abstain=("abstain", "mean")).reset_index()
    abl.to_csv(RES / "ablation_table.csv", index=False)

    # ---- вклад агентов
    co = pd.read_csv(RES / "agent_contributions.csv").groupby(["algo", "agent"]).delta_R.agg(["mean", "std"]).reset_index()
    co.to_csv(RES / "agent_contributions_summary.csv", index=False)

    # ---- вычислительная стоимость
    cost = meta[meta.tag == "main"].groupby("algo").agg(train_s=("train_s", "mean"), params=("params", "first"),
                                                        infer_ms=("infer_ms_per_track", "mean")).loc[ORDER]
    cost.to_csv(RES / "compute_cost.csv")

    # ---- графики
    lc = pd.read_csv(RES / "learning_curves.csv")
    lc = lc[(lc.tag == "main") & lc.val_R.notna() & (lc.algo != "Равные веса")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in ORDER[:-1]:
        g = lc[lc.algo == a].groupby("steps").val_R
        m, s = g.mean(), g.std()
        ax.plot(m.index / 1e6, m.values, label=a, lw=1.6 if a in (best, "PPO (1 агент)") else 1)
        ax.fill_between(m.index / 1e6, m - s, m + s, alpha=0.08)
    ew = out["val"].set_index("algo").loc["Равные веса", "R_mean"]
    ax.axhline(ew, color="black", ls="--", lw=1, label="Равные веса (val)")
    ax.set_ylim(-30, 10)
    ax.set_xlabel("переходов среды, млн")
    ax.set_ylabel("средняя награда за эпизод на val")
    ax.legend(ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "learning_curves.png", dpi=150); plt.close(fig)

    t = out["test"].set_index("algo").loc[ORDER]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    col = ["#A5A5A5" if a in ("PPO (1 агент)", "Равные веса") else "#4472C4" for a in t.index]
    ax.bar(t.index, t.R_mean, color=col, yerr=[t.R_mean - t.R_mean_lo, t.R_mean_hi - t.R_mean], capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ценность разметки за эпизод (50 треков)")
    ax.set_title("MTAT test: среднее по 5 seed, 95% ДИ", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "test_comparison.png", dpi=150); plt.close(fig)

    # «кривая капитала»: накопленная ценность разметки по 256 эпизодам теста (seed 0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for a in [best, "PPO (1 агент)", "Равные веса", "IPPO" if best != "IPPO" else "MAPPO"]:
        ax.plot(np.cumsum(np.load(RES / "episode_R" / f"{a}_0.npy")), label=a)
    ax.set_xlabel("эпизод (50 треков)")
    ax.set_ylabel("накопленная ценность разметки")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "equity_curve.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 250)
    for s in ["test", "jamendo_2010_2017"]:
        print("==", s)
        print(out[s][["algo", "R_mean", "R_mean_lo", "R_mean_hi", "sharpe", "cvar5", "max_drawdown", "catalog_acc",
                      "auto_err", "review_rate", "violations", "sla_breach", "abstain", "R_seed_std"]].round(3).to_string(index=False))
    print(pd.DataFrame(pair).round(4).to_string(index=False))
    print(st.loc[ORDER].round(2).to_string())
    print(abl.round(3).to_string(index=False))
    print(co.round(2).to_string(index=False))
    print(cost.round(3).to_string())
    print("best MARL by val:", best)


if __name__ == "__main__":
    main()
