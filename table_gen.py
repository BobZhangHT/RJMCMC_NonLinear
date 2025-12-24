#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a LaTeX simulation table (Table 1 style) from simu_summary.csv.

Column mapping (as requested):
- ASE  <- M_SE
- CP   <- CR
- Bias <- Bias
- ESD  <- ESD
- Subscript 1 -> beta_1, subscript 2 -> beta_2

Rounding rules (as requested):
- Bias / ESD / ASE: 3 decimals
- CP: 2 decimals
- H, K: 1 decimal

Output uses \\resizebox{\\textwidth}{!}{...} to fit page text width.
"""

import argparse
from pathlib import Path

import pandas as pd


def fmt(x, ndigits: int) -> str:
    """Fixed-decimal formatter with standard rounding."""
    return f"{float(x):.{ndigits}f}"


def normalize_g_type(s: str) -> str:
    """
    Normalize G_type values to canonical names:
    - linear
    - quadratic
    - sinusoidal

    Your CSV uses: linear / quad / sin
    """
    s0 = str(s).strip().lower()

    g_alias = {
        "linear": "linear",
        "lin": "linear",

        "quad": "quadratic",
        "quadratic": "quadratic",
        "quadr": "quadratic",

        "sin": "sinusoidal",
        "sine": "sinusoidal",
        "sinusoidal": "sinusoidal",
        "sinusoid": "sinusoidal",
    }
    return g_alias.get(s0, s0)


def normalize_method(s: str) -> str:
    """Normalize Method values to canonical internal keys."""
    s0 = str(s).strip()
    m_alias = {
        "NonLinear1": "NonLinear1",
        "NonLinear2": "NonLinear2",
        "CoxPH": "CoxPH",
        "Cox PH": "CoxPH",
        "CoxPH ": "CoxPH",
    }
    return m_alias.get(s0, s0)


def build_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    required = {
        "G_type", "N", "Method",
        "Bias1", "M_SE1", "ESD1", "CR1",
        "Bias2", "M_SE2", "ESD2", "CR2",
        "H", "K"
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Display labels in the final table
    g_display = {
        "linear": "Linear",
        "quad": "Quadratic",
        "quadratic": "Quadratic",
        "sin": "Sinusoidal",
        "sinusoidal": "Sinusoidal",
    }
    method_display = {"NonLinear1": "NonLinear1", "NonLinear2": "NonLinear2", "CoxPH": "Cox PH"}

    # Canonical ordering (linear -> quad -> sin)
    g_order_pref = ["linear", "quadratic", "sinusoidal"]
    method_order = ["NonLinear1", "NonLinear2", "CoxPH"]
    n_order_pref = [200, 400, 800]

    # Normalize keys
    df = df.copy()
    df["G_type"] = df["G_type"].apply(normalize_g_type)
    df["Method"] = df["Method"].apply(normalize_method)

    # Build ordering from data (never drop rows)
    g_present = [g for g in g_order_pref if g in df["G_type"].unique()]
    if not g_present:
        g_present = sorted(df["G_type"].unique())
    unknown_g = sorted(set(df["G_type"]) - set(g_present))
    g_order = g_present + unknown_g

    unknown_m = sorted(set(df["Method"]) - set(method_order))
    if unknown_m:
        method_order = method_order + unknown_m

    n_present = [n for n in n_order_pref if n in df["N"].astype(int).unique()]
    if not n_present:
        n_present = sorted(df["N"].astype(int).unique())
    unknown_n = sorted(set(df["N"].astype(int)) - set(n_present))
    n_order = n_present + unknown_n

    # Enforce ordering
    df["G_type"] = pd.Categorical(df["G_type"], categories=g_order, ordered=True)
    df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)
    df["N"] = pd.Categorical(df["N"].astype(int), categories=n_order, ordered=True)

    df = df.sort_values(["G_type", "Method", "N"]).reset_index(drop=True)

    # Start LaTeX
    lines = []
    lines.append(r"% Requires: \usepackage{booktabs}, \usepackage{multirow}, \usepackage{graphicx}")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append("")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llr rr rrrr rrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{$g(z)$} & \multirow{2}{*}{Method} & \multirow{2}{*}{$n$} & "
        r"\multirow{2}{*}{$H$} & \multirow{2}{*}{$K$} & "
        r"\multicolumn{4}{c}{$\beta_1$} & \multicolumn{4}{c}{$\beta_2$} \\"
    )
    lines.append(r"\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(r"& & & & & Bias & ESD & ASE & CP & Bias & ESD & ASE & CP \\")
    lines.append(r"\midrule")

    # Build body with multirow blocks
    for g in g_order:
        df_g = df[df["G_type"] == g]
        if df_g.empty:
            continue

        g_label = g_display.get(g, str(g))
        g_span = len(df_g)

        first_row_in_g = True

        # Determine which methods exist under this g
        methods_in_g = [m for m in method_order if not df_g[df_g["Method"] == m].empty]

        for m in methods_in_g:
            df_gm = df_g[df_g["Method"] == m]
            m_label = method_display.get(m, str(m))
            m_span = len(df_gm)

            for r_i, (_, row) in enumerate(df_gm.iterrows()):
                # Multirow labels
                if first_row_in_g:
                    g_cell = rf"\multirow{{{g_span}}}{{*}}{{{g_label}}}"
                    first_row_in_g = False
                else:
                    g_cell = ""

                if r_i == 0:
                    m_cell = rf"\multirow{{{m_span}}}{{*}}{{{m_label}}}"
                else:
                    m_cell = ""

                n = int(row["N"])
                H = fmt(row["H"], 1)
                K = fmt(row["K"], 1)

                # beta_1: Bias1, ESD1, M_SE1->ASE, CR1->CP
                b1_bias = fmt(row["Bias1"], 3)
                b1_esd = fmt(row["ESD1"], 3)
                b1_ase = fmt(row["M_SE1"], 3)
                b1_cp = fmt(row["CR1"], 2)

                # beta_2: Bias2, ESD2, M_SE2->ASE, CR2->CP
                b2_bias = fmt(row["Bias2"], 3)
                b2_esd = fmt(row["ESD2"], 3)
                b2_ase = fmt(row["M_SE2"], 3)
                b2_cp = fmt(row["CR2"], 2)

                lines.append(
                    f"{g_cell} & {m_cell} & {n} & {H} & {K} & "
                    f"{b1_bias} & {b1_esd} & {b1_ase} & {b1_cp} & "
                    f"{b2_bias} & {b2_esd} & {b2_ase} & {b2_cp} \\\\"
                )

            # Method separator inside a g-block
            if m != methods_in_g[-1]:
                lines.append(r"\cmidrule(lr){2-13}")

        # Separator between g-blocks
        g_existing = [gg for gg in g_order if not df[df["G_type"] == gg].empty]
        if g != g_existing[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser(
        description="Convert simu_summary.csv into a LaTeX table with resizebox(textwidth)."
    )
    parser.add_argument(
        "--mode", choices=["demo", "full"], default="full",
        help="Which simulation directory to read from (default: full)."
    )
    parser.add_argument(
        "--input", "-i", default="simu_summary.csv",
        help="Path to simu_summary.csv"
    )
    parser.add_argument(
        "--output", "-o", default="table_sim_results.tex",
        help="Output .tex file path"
    )
    parser.add_argument(
        "--caption", default=("Simulation results showing the bias, empirical standard deviation (ESD), "
                              "averaged standard error (ASE), and coverage probability (CP) of the 95\\% "
                              "credible intervals for $\\beta_1$ and $\\beta_2$ based on the proposed "
                              "NonLinear1, NonLinear2 and Cox PH models. Under each combination of sample "
                              "sizes $n$ and functional forms of $g$, 1{,}000 replications are used for reporting."),
        help="LaTeX caption text"
    )
    parser.add_argument(
        "--label", default="tab:sim_results",
        help="LaTeX label"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.input == "simu_summary.csv":
        input_path = Path("results") / "simulation" / args.mode / "simu_summary.csv"
    df = pd.read_csv(input_path)
    tex = build_latex_table(df, caption=args.caption, label=args.label)

    output_path = Path(args.output)
    if args.output == "table_sim_results.tex":
        output_path = Path("results") / "simulation" / args.mode / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex + "\n")

    print(f"Read: {input_path}")
    print(f"Wrote LaTeX table to: {output_path}")


if __name__ == "__main__":
    main()
