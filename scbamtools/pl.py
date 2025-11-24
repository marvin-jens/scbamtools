import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def edit_stats(df, query="query", ref="ref"):

    def extract_pos(x):
        if M := re.search(r"\d+", x[1:]):
            return int(M.group())
        else:
            return -1

    if not "op" in df.columns:
        df["op"] = df["edit"].apply(lambda x: x[0])
        df["pos"] = df["edit"].apply(extract_pos)

    op = df.groupby("op")["n"].sum()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{query} vs {ref} barcode match statistics")

    #
    order = ["=", "S", "I", "_", "X"]
    labels = ["exact", "subst.", "ins", "del", "no match"]

    _ = ax1.pie(
        op.loc[order],
        labels=labels,
        colors=plt.color_sequences["tab20c"],
        autopct="%.2f%%",
        pctdistance=0.75,
        counterclock=False,
        startangle=180,
        wedgeprops=dict(width=0.5, edgecolor="w"),
    )

    S_freq = df.loc[df["op"] == "S"].groupby("pos")["n"].sum()
    I_freq = df.loc[df["op"] == "I"].groupby("pos")["n"].sum()
    D_freq = df.loc[df["op"] == "_"].groupby("pos")["n"].sum()

    # frequencies of edits at each barcode position
    N = S_freq.sum() + I_freq.sum() + D_freq.sum()

    ax2.plot(S_freq / N, label="subst.")
    ax2.plot(I_freq / N, label="ins")
    ax2.plot(D_freq / N, label="del")
    ax2.legend()
    ax2.set_xlabel("position [nt]")
    ax2.set_ylabel("fraction of edits")
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)

    # boost relative to exact matches
    f = df.groupby("op")["n"].agg("sum")
    F = f / f.sum()
    boost = F / F.loc["="]

    g = sns.barplot(100 * boost.loc[["S", "I", "_"]], orient="h", ax=ax3, width=0.5)
    g.patches[1].set_color("orange")
    g.patches[2].set_color("green")

    ax3.set_xlabel("increase relative to exact matches [%]")
    ax3.set_yticklabels(["mismatch", "ins", "del"])
    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)

    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", padding=10)

    fig.tight_layout()

    return fig
