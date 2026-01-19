import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def preprocessing_stats(df, sample_id="sample", show=True):
    df_pre = df.copy()

    # read level stats
    df_pre['count_M'] = df_pre['count'] / 1e6
    total_reads = float(df_pre.query("value == 'input' and measure == 'reads'")['count_M'].iloc[0])

    def norm_percent(x, total):
        p = 100 * x / total
        if p < 0.5:
            return ""
        else:
            return f'{p:.0f} %'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{sample_id} read pre-processing statistics")
    sns.barplot(df_pre.loc[df_pre['measure'] == 'reads'], x='value', y='count_M', hue='name', hue_order=['N', 'A5', 'A3'], ax=ax1)
    for cont in ax1.containers:
        ax1.bar_label(cont, fmt='%.2f')
        labels = [norm_percent(x, total_reads) for x in cont.datavalues]
        ax1.bar_label(cont, label_type='center', labels=labels)

    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.set_ylabel("reads [M]")
    _ = ax1.set_xlabel("pre-processing flags")

    # base level stats
    df_bases = df_pre.loc[df_pre['measure'] == 'bases']
    valid = df_bases['value'].str.isnumeric()
    df_bases = df_bases.loc[valid]
    df_bases['value'] = df_bases['value'].astype(float)
    df_bases['bases'] = df_bases['count'] * df_bases['value']

    bases_breakdown = df_bases[['name', 'value', 'bases']].groupby('name').agg('sum')
    bases_breakdown['giga_bases'] = bases_breakdown['bases'] / 1e9
    bases_breakdown = bases_breakdown.rename(index={'L_in': 'input', 'L_out': 'output'})
    total_bases = float(bases_breakdown.loc['input', 'giga_bases'])

    sns.barplot(
            data=bases_breakdown,
            x='name', y='giga_bases',
            ax=ax2
        )
    ax2.set_ylabel("bases [G]")

    for cont in ax2.containers:
        ax2.bar_label(cont, fmt='%.2f')
        labels = [norm_percent(x, total_bases) for x in cont.datavalues]
        ax2.bar_label(cont, label_type='center', labels=labels)

    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.set_xlabel("pre-processing flags")    

    xmin, xmax = df_bases['value'].min(), df_bases['value'].max()

    #ax3.title(f"{sample_id} length distributions")

    for name in df_bases['name'].unique():
        df_name = df_bases.loc[df_bases['name'] == name].sort_values('value')
        x = df_name['value']
        y = df_name['count']
        cdf = np.cumsum(y) / np.sum(y)
        steps = ax3.step(x, cdf, label=name, where='mid')
        if name == "L_out":
            ax3.axvline(x=18, color=steps[0].get_color(), linestyle='--', label='L_out cutoff')

        ax3.set_xlabel("length [nt]")
        ax3.set_ylabel("cumulative fraction")

    
    # ax3.set_yscale('log')
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks(np.arange(xmin, xmax+1, step=10))
    ax3.legend(ncols=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax3.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    
    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig


def edit_stats(df, query="query", ref="ref", show=True):
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
    ax3.set_yticks(np.arange(4))
    ax3.set_yticklabels(["combined", "mismatch", "ins", "del"])
    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)

    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", padding=10)

    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig
