"""Полный эксперимент ЛР3: 8 MARL + 2 baseline × 5 seed, оценка на тесте, отложенном периоде, стресс-тесты, абляции.

Все алгоритмы получают одинаковый бюджет: 600 итераций × 64 среды × 50 треков = 1,92 млн переходов.
Оценка — 256 эпизодов с одинаковым seed для всех алгоритмов (общие случайные числа -> парные сравнения).
"""
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import env as E
import marl

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STREAM = ROOT / "data" / "stream.parquet"
SEEDS = [0, 1, 2, 3, 4]
marl.HP.update({"iters": 600, "eval_every": 20})
N_EVAL = 256

SETS = {
    "val": ("val", None), "test": ("test", None),
    "jamendo_2010_2017": ("sdd", (2010, 2017)), "jamendo_2005_2009": ("sdd", (2005, 2009)),
}
DATA = {k: E.load_split(STREAM, s, yr) for k, (s, yr) in SETS.items()}
DATA["train"] = E.load_split(STREAM, "train")
BASE = E.EnvConfig()
STRESS = {  # оцениваются на MTAT test, политика обучена в базовых условиях
    "нет текста": replace(BASE, drop_modality="text"),
    "нет ряда": replace(BASE, drop_modality="ts"),
    "нет изображения": replace(BASE, drop_modality="img"),
    "бюджет проверки ×0,5": replace(BASE, budget=7),
    "ошибка дороже (−5)": replace(BASE, r_wrong=-5.0),
}


def summarize(ev, cfg):
    R = ev["R"]
    cum = np.cumsum(ev["step_r"].reshape(-1))
    dd = (cum - np.maximum.accumulate(cum)).min()
    auto_n = ev["correct"] + ev["wrong"]
    err_auto = ev["wrong"] / np.maximum(auto_n, 1e-9)
    q05 = np.quantile(R, 0.05)
    return {"R_mean": R.mean(), "R_std": R.std(), "sharpe": R.mean() / (R.std() + 1e-9),
            "sortino": R.mean() / (np.sqrt(np.mean(np.minimum(R, 0) ** 2)) + 1e-9),
            "cvar5": R[R <= q05].mean(), "max_drawdown": dd,
            "catalog_acc": ev["final_correct"].mean(), "auto_rate": auto_n.mean(),
            "auto_err": ev["wrong"].sum() / max(auto_n.sum(), 1e-9), "review_rate": ev["review"].mean(),
            "unlabeled": ev["unlabeled"].mean(), "violations": ev["violation"].mean(),
            "sla_breach": (err_auto > 0.10).mean(), "abstain": ev["abstain"].mean()}


def run_one(name, seed, cfg=BASE, tag="main"):
    alg = marl.ALGOS[name](seed)
    log = []
    t0 = time.perf_counter()
    best_val = alg.train(*DATA["train"], *DATA["val"], cfg, seed, log)
    train_s = time.perf_counter() - t0
    pol = alg.policy()
    rows, curves = [], [dict(r, algo=name, seed=seed, tag=tag) for r in log]
    for k in ["val", "test", "jamendo_2010_2017", "jamendo_2005_2009"]:
        ev = marl.evaluate(pol, *DATA[k], cfg, n_episodes=N_EVAL, seed=777)
        rows.append({"algo": name, "seed": seed, "tag": tag, "set": k, **summarize(ev, cfg)})
        if k == "test" and tag == "main":
            np.save(RES / "episode_R" / f"{name}_{seed}.npy", ev["R"])
    if tag == "main":
        for sname, scfg in STRESS.items():
            ev = marl.evaluate(pol, *DATA["test"], scfg, n_episodes=N_EVAL, seed=777)
            rows.append({"algo": name, "seed": seed, "tag": tag, "set": f"stress: {sname}", **summarize(ev, scfg)})
    t1 = time.perf_counter()
    marl.evaluate(pol, *DATA["test"], cfg, n_episodes=N_EVAL, seed=777)
    infer_ms = (time.perf_counter() - t1) / (N_EVAL * cfg.T) * 1000
    meta = {"algo": name, "seed": seed, "tag": tag, "train_s": train_s, "best_val_R": float(best_val),
            "params": marl.n_params(alg), "infer_ms_per_track": infer_ms}
    return rows, curves, meta, alg


def contributions(alg, name, seed):
    """Вклад агентов: заменяем действия агента на «воздержаться» (эксперт) или «всегда автометка» (координатор)."""
    base_pol = alg.policy()
    out = []
    for i, who in enumerate(["эксперт: текст", "эксперт: ряд", "эксперт: изображение", "координатор"]):
        def pol(o, s, m, i=i):
            a = base_pol(o, s, m)
            a[:, i] = 4 if i < 3 else 0
            return a
        ev_b = marl.evaluate(base_pol, *DATA["test"], BASE, n_episodes=N_EVAL, seed=777)
        ev_x = marl.evaluate(pol, *DATA["test"], BASE, n_episodes=N_EVAL, seed=777)
        out.append({"algo": name, "seed": seed, "agent": who, "delta_R": float(ev_b["R"].mean() - ev_x["R"].mean())})
    return out


def main():
    (RES / "episode_R").mkdir(parents=True, exist_ok=True)
    rows, curves, metas, contrib = [], [], [], []
    for name in marl.ALGOS:
        for seed in SEEDS:
            r, c, m, alg = run_one(name, seed)
            rows += r; curves += c; metas.append(m)
            if name not in ("Равные веса",):
                contrib += contributions(alg, name, seed)
            print(f"{name:14s} seed={seed} val={m['best_val_R']:7.2f} test={r[1]['R_mean']:7.2f} {m['train_s']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RES / "eval_all.csv", index=False)
    pd.DataFrame(curves).to_csv(RES / "learning_curves.csv", index=False)
    pd.DataFrame(metas).to_csv(RES / "runs_meta.csv", index=False)
    pd.DataFrame(contrib).to_csv(RES / "agent_contributions.csv", index=False)

    # выбор лучшего MARL по val (не по тесту!) -> абляции
    ev = pd.DataFrame(rows)
    marl_only = ev[(ev.set == "val") & ~ev.algo.isin(["PPO (1 агент)", "Равные веса"])]
    best = marl_only.groupby("algo").R_mean.mean().idxmax()
    abl_rows = []
    for aname, acfg in {"без координатора": replace(BASE, use_coordinator=False),
                        "без воздержания экспертов": replace(BASE, allow_abstain=False),
                        "без репутации (обратной связи)": replace(BASE, use_trust=False)}.items():
        for seed in SEEDS:
            r, c, m, _ = run_one(best, seed, cfg=acfg, tag=aname)
            abl_rows += r; curves += c; metas.append(m)
            print(f"ablation {aname} seed={seed} test={r[1]['R_mean']:.2f}", flush=True)
    pd.DataFrame(abl_rows).to_csv(RES / "ablations.csv", index=False)
    json.dump({"hyperparameters": marl.HP, "seeds": SEEDS, "env": asdict(BASE), "n_eval_episodes": N_EVAL,
               "best_marl_by_val": best, "stress": {k: asdict(v) for k, v in STRESS.items()}},
              open(RES / "config_log.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
