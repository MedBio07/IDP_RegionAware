#!/usr/bin/env python3
"""Create FusionEncoder-style benchmark tables and a method overview figure."""

from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("SL329", "MXD494", "DISORDER723")
P4_8_VARIANT = "p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble"
TABLE_DIR = ROOT / "manuscript/tables"
LATEX_TABLE_DIR = ROOT / "manuscript/latex/bioinformatics/tables"
FIGURE_DIR = ROOT / "figures/p5_8"
LATEX_FIG_DIR = ROOT / "manuscript/latex/bioinformatics/Fig"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: object) -> float:
    if value in (None, "", "NA"):
        return math.nan
    return float(value)


def fmt3(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def fmt6(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return str(value)


def row_by(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in criteria.items())
        raise ValueError(f"expected one row for {detail}; found {len(matches)}")
    return matches[0]


def parse_literature_markdown(path: Path) -> dict[str, list[dict[str, object]]]:
    rows_by_dataset: dict[str, list[dict[str, object]]] = {dataset: [] for dataset in DATASETS}
    current_dataset: str | None = None
    section_re = re.compile(r"^##\s+(.+?)\s*$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        section_match = section_re.match(raw_line)
        if section_match:
            name = section_match.group(1).strip()
            current_dataset = name if name in rows_by_dataset else None
            continue
        if current_dataset is None or not raw_line.startswith("|"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) < 8 or cells[0] in {"Method", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        rows_by_dataset[current_dataset].append(
            {
                "dataset": current_dataset,
                "predictor": cells[0],
                "year": cells[1],
                "sn": as_float(cells[2]),
                "sp": as_float(cells[3]),
                "bacc": as_float(cells[4]),
                "mcc": as_float(cells[5]),
                "auc": as_float(cells[6]),
                "source": cells[7],
                "is_this_work": False,
            }
        )
    return rows_by_dataset


def add_this_work(rows_by_dataset: dict[str, list[dict[str, object]]]) -> None:
    benchmark_rows = read_tsv(ROOT / "results/p4_8/P4_8_MAIN_BENCHMARK_REPLACEMENT.tsv")
    for dataset in DATASETS:
        candidate = row_by(benchmark_rows, dataset=dataset, variant=P4_8_VARIANT, method="platt")
        rows_by_dataset[dataset].append(
            {
                "dataset": dataset,
                "predictor": "RegionAdapterMoETCN (this work)",
                "year": "2026",
                "sn": as_float(candidate["sn"]),
                "sp": as_float(candidate["sp"]),
                "bacc": as_float(candidate["bacc"]),
                "mcc": as_float(candidate["mcc"]),
                "auc": as_float(candidate["auc"]),
                "source": "This work: P4.8 full DM3000, 3-seed ensemble, Platt",
                "is_this_work": True,
            }
        )


def add_competition_ranks(rows: list[dict[str, object]]) -> None:
    for metric in ("auc", "bacc", "mcc"):
        for row in rows:
            value = as_float(row[metric])
            row[f"rank_{metric}"] = 1 + sum(as_float(other[metric]) > value + 1.0e-12 for other in rows)


def ranked_rows(rows_by_dataset: dict[str, list[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    ranked: dict[str, list[dict[str, object]]] = {}
    for dataset, rows in rows_by_dataset.items():
        add_competition_ranks(rows)
        ranked[dataset] = sorted(rows, key=lambda row: (-as_float(row["auc"]), -as_float(row["bacc"]), -as_float(row["mcc"]), str(row["predictor"])))
    return ranked


def main_rows_for_latex(ranked: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        selected = [row for row in ranked[dataset] if int(row["rank_auc"]) <= 5]
        rows.extend(selected)
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["dataset", "predictor", "year", "sn", "sp", "bacc", "mcc", "auc", "rank_auc", "rank_bacc", "rank_mcc", "source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: fmt6(row.get(field, ""))
                    if field in {"sn", "sp", "bacc", "mcc", "auc"}
                    else str(row.get(field, ""))
                    for field in fields
                }
            )


def md_metric(row: dict[str, object], metric: str) -> str:
    value = fmt3(row[metric])
    return f"**{value}**" if int(row[f"rank_{metric}"]) == 1 else value


def md_rank(row: dict[str, object], metric: str) -> str:
    value = str(row[f"rank_{metric}"])
    return f"**{value}**" if int(row[f"rank_{metric}"]) == 1 else value


def write_markdown(path: Path, title: str, ranked: dict[str, list[dict[str, object]]], top_only: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for dataset in DATASETS:
        source_rows = [row for row in ranked[dataset] if int(row["rank_auc"]) <= 5] if top_only else ranked[dataset]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Predictor | Sn | Sp | BACC | MCC | AUC | Rank AUC | Rank BACC | Rank MCC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in source_rows:
            predictor = f"**{row['predictor']}**" if row["is_this_work"] else str(row["predictor"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        predictor,
                        fmt3(row["sn"]),
                        fmt3(row["sp"]),
                        md_metric(row, "bacc"),
                        md_metric(row, "mcc"),
                        md_metric(row, "auc"),
                        md_rank(row, "auc"),
                        md_rank(row, "bacc"),
                        md_rank(row, "mcc"),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append(
        "Note: ranks follow the FusionEncoder convention and are computed within each dataset after adding this work to the curated local direct-benchmark table. Higher is better for AUC, BACC and MCC."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_latex(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def tex_metric(row: dict[str, object], metric: str) -> str:
    value = fmt3(row[metric])
    return rf"\textbf{{{value}}}" if int(row[f"rank_{metric}"]) == 1 else value


def tex_rank(row: dict[str, object], metric: str) -> str:
    value = str(row[f"rank_{metric}"])
    return rf"\textbf{{{value}}}" if int(row[f"rank_{metric}"]) == 1 else value


def tex_predictor(row: dict[str, object]) -> str:
    if row["is_this_work"]:
        return r"\textbf{\method{} (this work)}"
    return escape_latex(row["predictor"])


def write_latex_table(path: Path, ranked: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{FusionEncoder-style independent benchmark comparison. Rows are sorted by AUC within each test dataset; the displayed rows include this work and all methods with AUC rank up to 5 after adding this work to the curated direct-benchmark table. Binary metrics use the DM1229 validation-selected threshold for this work. External methods are published aggregate point references and are not paired statistical comparisons.}",
        r"\label{tab:full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Dataset & Predictor & Sn & Sp & BACC & MCC & AUC & \multicolumn{3}{c}{Rank} \\",
        r"\cmidrule(lr){8-10}",
        r" & & & & & & & AUC & BACC & MCC \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        if dataset_index:
            lines.append(r"\midrule")
        selected = [row for row in ranked[dataset] if int(row["rank_auc"]) <= 5]
        for row_index, row in enumerate(selected):
            dataset_cell = escape_latex(dataset) if row_index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        dataset_cell,
                        tex_predictor(row),
                        fmt3(row["sn"]),
                        fmt3(row["sp"]),
                        tex_metric(row, "bacc"),
                        tex_metric(row, "mcc"),
                        tex_metric(row, "auc"),
                        tex_rank(row, "auc"),
                        tex_rank(row, "bacc"),
                        tex_rank(row, "mcc"),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_method_overview(path_pdf: Path, path_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        face: str = "#f8fafc",
        edge: str = "#334155",
        fontsize: float = 9.0,
        weight: str = "normal",
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#0f172a", fontweight=weight)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = "#475569", curve: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.1,
                color=color,
                connectionstyle=f"arc3,rad={curve}",
            )
        )

    ax.text(0.50, 0.965, "RegionAdapterMoETCN: sequence-only region-adapter mixture of experts", ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    panel_color = "#475569"
    ax.text(0.06, 0.895, "(a) Residue features", ha="left", va="center", fontsize=10, fontweight="bold", color=panel_color)
    ax.text(0.29, 0.895, "(b) Warm-start backbone", ha="left", va="center", fontsize=10, fontweight="bold", color=panel_color)
    ax.text(0.53, 0.895, "(c) Region-adapter MoE", ha="left", va="center", fontsize=10, fontweight="bold", color=panel_color)
    ax.text(0.77, 0.895, "(d) Calibration and output", ha="left", va="center", fontsize=10, fontweight="bold", color=panel_color)

    box(0.035, 0.58, 0.14, 0.14, "Protein\nsequence", face="#eef2ff", edge="#4f46e5", fontsize=10, weight="bold")
    box(0.215, 0.74, 0.17, 0.10, "Frozen ESM2-t33\nresidue embedding", face="#eff6ff", edge="#2563eb")
    box(0.215, 0.58, 0.17, 0.10, "Amino-acid\none-hot", face="#ecfdf5", edge="#059669")
    box(0.215, 0.42, 0.17, 0.10, "Relative\nposition", face="#fff7ed", edge="#ea580c")
    box(0.425, 0.58, 0.15, 0.14, "Frozen shared\nTCN blocks", face="#f8fafc", edge="#334155", fontsize=10, weight="bold")
    box(0.405, 0.78, 0.19, 0.07, "P4.6 RegionAwareTCN checkpoint", face="#f1f5f9", edge="#64748b", fontsize=8.7)

    adapters = [
        ("SDR adapter\n+ expert head", 0.625, 0.765, "#e0f2fe", "#0284c7"),
        ("LDR adapter\n+ expert head", 0.625, 0.635, "#dcfce7", "#16a34a"),
        ("Terminal-IDR adapter\n+ expert head", 0.625, 0.505, "#fef3c7", "#d97706"),
        ("Internal-IDR adapter\n+ expert head", 0.625, 0.375, "#fee2e2", "#dc2626"),
    ]
    for label, x, y, face, edge in adapters:
        box(x, y, 0.16, 0.085, label, face=face, edge=edge, fontsize=8.3)
    box(0.625, 0.20, 0.16, 0.09, "Residue-level\nsoftmax gate", face="#f5f3ff", edge="#7c3aed", fontsize=8.8, weight="bold")
    box(0.815, 0.51, 0.10, 0.12, "Expert\nmixture\nlogit", face="#f8fafc", edge="#334155", fontsize=9.0, weight="bold")
    box(0.815, 0.30, 0.10, 0.09, "Generic\ndisorder logit", face="#f1f5f9", edge="#64748b", fontsize=8.6)
    box(0.82, 0.74, 0.13, 0.09, "3-seed score\naveraging", face="#ecfeff", edge="#0891b2", fontsize=8.6)
    box(0.82, 0.15, 0.13, 0.09, "Platt calibration\nDM1229 only", face="#fefce8", edge="#ca8a04", fontsize=8.6)
    box(0.835, 0.015, 0.14, 0.10, "Calibrated IDR\nprobability + entropy\nuncertainty", face="#f8fafc", edge="#0f172a", fontsize=8.3, weight="bold")

    arrow((0.175, 0.65), (0.215, 0.79), curve=0.15)
    arrow((0.175, 0.65), (0.215, 0.63))
    arrow((0.175, 0.65), (0.215, 0.47), curve=-0.15)
    arrow((0.385, 0.79), (0.425, 0.67), curve=-0.18)
    arrow((0.385, 0.63), (0.425, 0.65))
    arrow((0.385, 0.47), (0.425, 0.62), curve=0.18)
    arrow((0.50, 0.78), (0.50, 0.72), color="#64748b")
    for _, _, y, _, edge in adapters:
        arrow((0.575, 0.65), (0.625, y + 0.042), color=edge)
        arrow((0.785, y + 0.042), (0.815, 0.57), color=edge)
    arrow((0.575, 0.61), (0.625, 0.245), color="#7c3aed", curve=-0.25)
    arrow((0.785, 0.245), (0.815, 0.545), color="#7c3aed", curve=0.25)
    arrow((0.865, 0.51), (0.865, 0.39), color="#475569")
    arrow((0.865, 0.63), (0.885, 0.74), color="#0891b2")
    arrow((0.885, 0.74), (0.885, 0.24), color="#0891b2", curve=-0.12)
    arrow((0.885, 0.30), (0.885, 0.24), color="#64748b")
    arrow((0.885, 0.15), (0.895, 0.115), color="#ca8a04")

    ax.text(0.105, 0.30, "No PSSM, MSA,\nPDB coordinates,\nor AlphaFold inputs", ha="center", va="center", fontsize=8.5, color="#334155")
    ax.text(0.505, 0.30, "Only adapters,\nexpert heads,\nand gate are trained", ha="center", va="center", fontsize=8.5, color="#334155")
    ax.text(0.705, 0.125, "Gate weights are analyzed\nas auxiliary mechanism evidence", ha="center", va="center", fontsize=8.1, color="#334155")

    fig.tight_layout(pad=0.7)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def main() -> None:
    rows_by_dataset = parse_literature_markdown(ROOT / "results/literature_test_results/current_methods_with_year_results.md")
    add_this_work(rows_by_dataset)
    ranked = ranked_rows(rows_by_dataset)
    full_rows = [row for dataset in DATASETS for row in ranked[dataset]]
    main_rows = main_rows_for_latex(ranked)

    write_tsv(TABLE_DIR / "P5_8_Table2_FusionEncoder_style_main_comparison.tsv", main_rows)
    write_tsv(TABLE_DIR / "P5_8_TableS_FusionEncoder_style_full_rankings.tsv", full_rows)
    write_markdown(
        TABLE_DIR / "P5_8_Table2_FusionEncoder_style_main_comparison.md",
        "P5.8 Table 2. FusionEncoder-style main benchmark comparison",
        ranked,
        top_only=True,
    )
    write_markdown(
        TABLE_DIR / "P5_8_TableS_FusionEncoder_style_full_rankings.md",
        "P5.8 Supplementary Table. FusionEncoder-style full benchmark rankings",
        ranked,
        top_only=False,
    )
    write_latex_table(LATEX_TABLE_DIR / "table2_fusionencoder_style_main.tex", ranked)

    method_pdf = FIGURE_DIR / "P5_8_REGIONADAPTERMOETCN_METHOD_OVERVIEW.pdf"
    method_png = FIGURE_DIR / "P5_8_REGIONADAPTERMOETCN_METHOD_OVERVIEW.png"
    plot_method_overview(method_pdf, method_png)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(method_pdf, LATEX_FIG_DIR / "figure1_method_overview.pdf")


if __name__ == "__main__":
    main()
