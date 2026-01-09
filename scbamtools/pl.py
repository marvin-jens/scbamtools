import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def edit_stats(df, query="query", ref="ref"):
    import scbamtools.tk as tk

    op, (S_freq, I_freq, D_freq) = tk.summarize_edit_stats(df)

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
    F.loc["combined"] = F.loc[["S", "I", "_"]].sum()
    boost = F / F.loc["="]
    # print(boost)

    g = sns.barplot(
        100 * boost.loc[["combined", "S", "I", "_"]], orient="h", ax=ax3, width=0.5
    )
    g.patches[0].set_color("goldenrod")
    g.patches[1].set_color("blue")
    g.patches[2].set_color("orange")
    g.patches[3].set_color("green")

    ax3.set_xlabel("increase relative to exact matches [%]")
    ax3.set_yticklabels(["combined", "mismatch", "ins", "del"])
    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)

    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", padding=10)

    fig.tight_layout()

    return fig
