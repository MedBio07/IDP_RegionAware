#!/usr/bin/env python3
"""Create FusionEncoder-style benchmark tables and a method overview figure."""

from __future__ import annotations

import csv
import json
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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    path_eps = path_pdf.with_suffix(".eps")
    path_svg = path_pdf.with_suffix(".svg")
    path_manifest = path_pdf.with_suffix(".manifest.json")
    palette = {
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "green": "#009E73",
        "purple": "#CC79A7",
        "black": "#111827",
        "gray": "#4B5563",
        "light_gray": "#F3F4F6",
        "pale_blue": "#EAF3FA",
        "pale_orange": "#FFF2E6",
        "pale_green": "#EAF7F1",
        "pale_purple": "#F8ECF4",
        "pale_yellow": "#FFF8E1",
        "white": "#FFFFFF",
    }

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    width_mm = 183.0
    height_mm = 108.0
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), layout="constrained")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        def box(
            x: float,
            y: float,
            w: float,
            h: float,
            text: str,
            face: str,
            edge: str,
            fontsize: float = 6.8,
            weight: str = "normal",
            linestyle: str = "-",
            linewidth: float = 0.9,
        ) -> None:
            patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.010",
                linewidth=linewidth,
                linestyle=linestyle,
                edgecolor=edge,
                facecolor=face,
                mutation_aspect=1.0,
            )
            ax.add_patch(patch)
            ax.text(
                x + w / 2,
                y + h / 2,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=palette["black"],
                fontweight=weight,
                linespacing=1.05,
            )

        def arrow(
            start: tuple[float, float],
            end: tuple[float, float],
            color: str = palette["gray"],
            curve: float = 0.0,
            linestyle: str = "-",
            linewidth: float = 0.85,
            mutation_scale: float = 8.0,
        ) -> None:
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=mutation_scale,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                    shrinkA=0.0,
                    shrinkB=0.0,
                    connectionstyle=f"arc3,rad={curve}",
                )
            )

        def panel_label(x: float, text: str) -> None:
            ax.text(x, 0.895, text, ha="left", va="center", fontsize=7.2, color=palette["black"], fontweight="bold")

        ax.text(
            0.50,
            0.962,
            "RegionAdapterMoETCN sequence-only prediction workflow",
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=palette["black"],
        )
        panel_label(0.026, "a  Residue inputs")
        panel_label(0.346, "b  Warm-start backbone")
        panel_label(0.612, "c  Region-adapter MoE")
        panel_label(0.850, "d  Calibration output")

        box(0.028, 0.520, 0.104, 0.120, "Protein\nsequence", palette["pale_blue"], palette["blue"], 7.0, "bold")
        feature_boxes = [
            ("Frozen ESM2-t33\nembedding", 0.180, 0.730, palette["pale_blue"], palette["blue"]),
            ("Amino-acid\none-hot", 0.180, 0.555, palette["pale_green"], palette["green"]),
            ("Relative\nposition", 0.180, 0.380, palette["pale_orange"], palette["vermillion"]),
        ]
        for label, x, y, face, edge in feature_boxes:
            box(x, y, 0.126, 0.100, label, face, edge, 6.2)
            arrow((0.132, 0.580), (x, y + 0.050), edge, curve=0.07 if y > 0.58 else -0.07)

        box(0.352, 0.540, 0.100, 0.105, "Concatenated\nresidue tensor", palette["white"], palette["black"], 6.4, "bold")
        for _, x, y, _, edge in feature_boxes:
            arrow((x + 0.126, y + 0.050), (0.352, 0.592), edge, curve=-0.02 if y > 0.58 else 0.02)

        box(0.492, 0.727, 0.130, 0.083, "P4.6 checkpoint\nloads weights", palette["light_gray"], palette["gray"], 5.9)
        box(0.492, 0.520, 0.130, 0.135, "Frozen shared\nTCN blocks", palette["light_gray"], palette["black"], 7.0, "bold")
        arrow((0.452, 0.592), (0.492, 0.592), palette["gray"])
        arrow((0.557, 0.727), (0.557, 0.655), palette["gray"], linewidth=0.8)

        adapter_specs = [
            ("SDR\nadapter", 0.682, 0.765, palette["pale_blue"], palette["blue"]),
            ("LDR\nadapter", 0.682, 0.632, palette["pale_green"], palette["green"]),
            ("Terminal-IDR\nadapter", 0.682, 0.499, palette["pale_orange"], palette["vermillion"]),
            ("Internal-IDR\nadapter", 0.682, 0.366, palette["pale_purple"], palette["purple"]),
        ]
        for label, x, y, face, edge in adapter_specs:
            box(x, y, 0.122, 0.078, label + "\n+ head", face, edge, 5.9, "bold")
            arrow((0.622, 0.588), (x, y + 0.039), edge, curve=0.07 if y > 0.58 else -0.07)

        box(0.682, 0.175, 0.122, 0.082, "Residue-level\ngate", palette["white"], palette["purple"], 6.4, "bold", linestyle="--")
        arrow((0.622, 0.548), (0.682, 0.216), palette["purple"], curve=-0.18, linestyle="--")

        box(0.845, 0.522, 0.072, 0.112, "Generic +\nweighted\nexpert mix", palette["white"], palette["black"], 5.6, "bold")
        for _, x, y, _, edge in adapter_specs:
            arrow((x + 0.122, y + 0.039), (0.845, 0.578), edge, curve=0.04 if y > 0.58 else -0.04)
        arrow((0.804, 0.216), (0.845, 0.548), palette["purple"], curve=0.22, linestyle="--")
        ax.text(0.735, 0.300, "Only adapters,\nheads and gate\nare trained", ha="center", va="center", fontsize=5.6, color=palette["gray"], linespacing=1.05)

        box(0.922, 0.700, 0.060, 0.076, "Average\n3 seeds", palette["pale_blue"], palette["blue"], 5.4)
        box(0.922, 0.410, 0.060, 0.076, "Platt\nscaling", palette["pale_yellow"], "#8A6D00", 5.4)
        box(0.913, 0.128, 0.076, 0.090, "Calibrated\nIDR p\n+ entropy", palette["white"], palette["black"], 5.3, "bold")
        arrow((0.917, 0.578), (0.952, 0.700), palette["blue"], curve=0.12)
        arrow((0.952, 0.700), (0.952, 0.486), palette["blue"], curve=-0.04)
        arrow((0.952, 0.410), (0.952, 0.218), "#8A6D00")

        ax.text(
            0.080,
            0.235,
            "Sequence-only inputs:\nno PSSM, MSA, PDB coordinates,\nAlphaFold confidence or function labels",
            ha="center",
            va="center",
            fontsize=5.8,
            color=palette["gray"],
            linespacing=1.1,
        )
        ax.text(
            0.430,
            0.225,
            "Frozen modules preserve the P4.6 backbone;\ntrainable adapters provide local score correction",
            ha="center",
            va="center",
            fontsize=5.8,
            color=palette["gray"],
            linespacing=1.1,
        )
        ax.text(
            0.743,
            0.094,
            "Gate weights are reported as auxiliary mechanism evidence,\nnot as direct biological region annotations",
            ha="center",
            va="center",
            fontsize=5.5,
            color=palette["gray"],
            linespacing=1.1,
        )

        fig.savefig(path_pdf, dpi=600, metadata={"Title": "RegionAdapterMoETCN method overview"})
        fig.savefig(path_eps, dpi=600)
        fig.savefig(path_svg, dpi=600)
        fig.savefig(path_png, dpi=600)
        plt.close(fig)

    manifest = {
        "figure": "P5.8 RegionAdapterMoETCN method overview",
        "audience_medium": "Bioinformatics manuscript, static full-width method figure",
        "figure_type": "workflow schematic; no quantitative data encoding",
        "target_size_mm": {"width": width_mm, "height": height_mm},
        "source_inputs": [
            "scripts/assemble_p5_8_fusionencoder_style_assets.py",
            "manuscript/latex/bioinformatics/main.tex",
        ],
        "transformations": [
            "manual schematic layout from the P4.8 architecture description",
            "no numeric data filtering, smoothing, normalization, or imputation",
        ],
        "accessibility": [
            "all module roles are labeled directly",
            "color is redundant with text labels and line styles",
            "Okabe-Ito-derived high-contrast category colors are used on white",
        ],
        "outputs": {
            "pdf": str(path_pdf.relative_to(ROOT)),
            "eps": str(path_eps.relative_to(ROOT)),
            "svg": str(path_svg.relative_to(ROOT)),
            "png_600dpi": str(path_png.relative_to(ROOT)),
        },
    }
    path_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
