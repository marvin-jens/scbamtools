import pandas as pd


def summarize_edit_stats(df):
    import re

    def extract_pos(x):
        if M := re.search(r"\d+", x[1:]):
            return int(M.group())
        else:
            return -1

    if not "op" in df.columns:
        df["op"] = df["edit"].apply(lambda x: x[0])
        df["pos"] = df["edit"].apply(extract_pos)

    op = df.groupby("op")["n"].sum()

    S_freq = df.loc[df["op"] == "S"].groupby("pos")["n"].sum()
    I_freq = df.loc[df["op"] == "I"].groupby("pos")["n"].sum()
    D_freq = df.loc[df["op"] == "_"].groupby("pos")["n"].sum()

    return op, (S_freq, I_freq, D_freq)
    # ov = pd.DataFrame({
    #     'obs', : ['N', 'N_edits', 'N_exact', 'f_edits', 'f_exact]'value'
    #     'N': df['n'].sum(),
    #     ''
    # })
