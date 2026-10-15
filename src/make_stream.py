"""Данные для среды: вероятности трёх модальных экспертов (из ЛР2) для каждого трека, без утечки.

- MTAT train: out-of-fold вероятности (GroupKFold=5 по исполнителям) — модели модальностей не видели трек;
- MTAT val / test: модели обучены на всём MTAT train;
- SDD (Jamendo): те же модели, обученные на MTAT train (перенос, гармонизированное аудио).
Выход: data/stream.parquet — по строке на трек: split, y, p_text[4], p_ts[4], p_img[4], метаданные.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
LAB2 = ROOT.parents[1] / "2" / "lab2_multimodal_vocal"
sys.path.insert(0, str(LAB2 / "src"))
import experiment as e2  # noqa: E402

C = {"text": 10, "ts": 10, "img": 10}  # C из журнала параметров ЛР2 (подобраны на val)


def fit(X, y, c):
    return LogisticRegression(C=c, max_iter=4000, class_weight="balanced").fit(X, y)


def main():
    df, seq, emb, _ = e2.load("mtat")
    sp = {s: (df.split == s).values for s in ["train", "val", "test"]}
    enc = e2.Encoders().fit(df[sp["train"]], seq[sp["train"]], emb[sp["train"]])
    F = {s: enc.transform(df[sp[s]], seq[sp[s]], emb[sp[s]]) for s in sp}
    y = {s: df.y.values[sp[s]] for s in sp}
    P = {s: {} for s in ["train", "val", "test", "sdd"]}
    # OOF на train
    groups = df.group_key.values[sp["train"]]
    for m, key in e2.UNI.items():
        oof = np.zeros((sp["train"].sum(), 4))
        for tr, va in GroupKFold(5).split(y["train"], y["train"], groups):
            oof[va] = fit(F["train"][key][tr], y["train"][tr], C[m]).predict_proba(F["train"][key][va])
        P["train"][m] = oof
        full = fit(F["train"][key], y["train"], C[m])
        P["val"][m] = full.predict_proba(F["val"][key])
        P["test"][m] = full.predict_proba(F["test"][key])
        if m == "text":
            text_full = full
    dfs, seqs, embs, _ = e2.load("sdd")
    Fs = enc.transform(dfs, seqs, embs)
    for m, key in e2.UNI.items():
        P["sdd"][m] = fit(F["train"][key], y["train"], C[m]).predict_proba(Fs[key])

    rows = []
    for s, d in [("train", df[sp["train"]]), ("val", df[sp["val"]]), ("test", df[sp["test"]]), ("sdd", dfs)]:
        out = pd.DataFrame({"id": d.id.values, "split": s, "y": d.y.values, "label": d.label.values,
                            "artist": d.group_key.astype(str).values,
                            "release_year": d.release_year.values if "release_year" in d else np.nan})
        for m in ["text", "ts", "img"]:
            for k in range(4):
                out[f"p_{m}_{k}"] = P[s][m][:, k]
        rows.append(out)
    st = pd.concat(rows, ignore_index=True)
    st.to_parquet(ROOT / "data" / "stream.parquet", index=False)
    for s, g in st.groupby("split"):
        acc = {m: (g[[f"p_{m}_{k}" for k in range(4)]].values.argmax(1) == g.y).mean().round(3) for m in ["text", "ts", "img"]}
        print(s, len(g), acc)


if __name__ == "__main__":
    main()
