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
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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
    height_mm = 156.0
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
            radius: float = 0.008,
        ) -> None:
            patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0.006,rounding_size={radius}",
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

        def panel(x: float, y: float, w: float, h: float, label: str, title: str) -> None:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#1F354D", linewidth=0.95))
            ax.text(x + 0.012, y + h - 0.020, f"({label})", ha="left", va="center", fontsize=7.8, fontweight="bold")
            ax.text(x + 0.050, y + h - 0.020, title, ha="left", va="center", fontsize=7.6, fontweight="bold")

        def inset(x: float, y: float, w: float, h: float, title: str, edge: str) -> None:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor=edge, linewidth=0.75, linestyle="--"))
            ax.text(x + 0.010, y + h - 0.018, title, fontsize=6.0, ha="left", va="center")

        def token_track(x: float, y: float, tokens: str, cell_w: float, cell_h: float) -> None:
            colors = [palette["pale_blue"], palette["pale_green"], palette["pale_orange"], palette["pale_purple"]]
            edges = [palette["blue"], palette["green"], palette["vermillion"], palette["purple"]]
            for index, token in enumerate(tokens):
                cx = x + index * cell_w
                ax.add_patch(
                    FancyBboxPatch(
                        (cx, y),
                        cell_w * 0.83,
                        cell_h,
                        boxstyle="round,pad=0.002,rounding_size=0.004",
                        facecolor=colors[index % len(colors)],
                        edgecolor=edges[index % len(edges)],
                        linewidth=0.7,
                    )
                )
                ax.text(cx + cell_w * 0.415, y + cell_h / 2, token, ha="center", va="center", fontsize=6.0, fontweight="bold")

        def matrix(x: float, y: float, rows: int, cols: int, w: float, h: float, edge: str, mode: str) -> None:
            cell_w = w / cols
            cell_h = h / rows
            for row in range(rows):
                for col in range(cols):
                    if mode == "esm":
                        intensity = (row * 2 + col * 3) % 7 / 6.0
                        face = mpl.colors.to_hex((0.88 - 0.35 * intensity, 0.95 - 0.18 * intensity, 0.99))
                    elif mode == "onehot":
                        face = palette["green"] if row == (col * 2 + 1) % rows else "#FFFFFF"
                    elif mode == "position":
                        intensity = col / max(cols - 1, 1)
                        face = mpl.colors.to_hex((1.0, 0.95 - 0.28 * intensity, 0.86 - 0.24 * intensity))
                    else:
                        face = "#FFFFFF"
                    ax.add_patch(Rectangle((x + col * cell_w, y + (rows - row - 1) * cell_h), cell_w, cell_h, facecolor=face, edgecolor=edge, linewidth=0.22))
            ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=edge, linewidth=0.7))

        def vector_strip(x: float, y: float, n: int, w: float, h: float, edge: str, face: str) -> None:
            gap = w * 0.006
            cell_w = (w - gap * (n - 1)) / n
            for index in range(n):
                ax.add_patch(Rectangle((x + index * (cell_w + gap), y), cell_w, h, facecolor=face, edgecolor=edge, linewidth=0.45))

        def conv_block(x: float, y: float, w: float, h: float, dilation: int) -> None:
            box(x, y, w, h, "", "#FFFFFF", palette["gray"], linewidth=0.65, radius=0.006)
            ax.text(x + w * 0.10, y + h * 0.78, f"d={dilation}", fontsize=4.8, color=palette["gray"], ha="left", va="center")
            ys = [y + h * 0.62, y + h * 0.42, y + h * 0.22]
            labels = ["k3", "k7", "k15"]
            for yy, lab in zip(ys, labels):
                ax.plot([x + w * 0.20, x + w * 0.58], [yy, yy], color=palette["blue"], linewidth=0.8)
                for tick in range(3):
                    ax.add_patch(Rectangle((x + w * (0.25 + tick * 0.11), yy - h * 0.035), w * 0.035, h * 0.07, facecolor=palette["pale_blue"], edgecolor=palette["blue"], linewidth=0.35))
                ax.text(x + w * 0.66, yy, lab, ha="left", va="center", fontsize=4.6)
            ax.text(x + w * 0.86, y + h * 0.45, "+", ha="center", va="center", fontsize=7.0, fontweight="bold")

        def adapter_block(x: float, y: float, w: float, h: float, label: str, edge: str, face: str) -> None:
            box(x, y, w, h, "", face, edge, linewidth=0.85, radius=0.006)
            ax.text(x + w * 0.08, y + h * 0.72, label, ha="left", va="center", fontsize=5.8, fontweight="bold")
            inner_y = y + h * 0.28
            parts = [("Down\n128->32", 0.08), ("GELU", 0.38), ("Up\n32->128", 0.62), ("Head\nlogit", 0.84)]
            for part, frac in parts:
                bw = w * (0.21 if "->" in part else 0.14)
                bx = x + w * frac
                box(bx, inner_y, bw, h * 0.28, part, "#FFFFFF", edge, fontsize=4.4, linewidth=0.45, radius=0.004)
            for start, end in [(0.29, 0.38), (0.52, 0.62), (0.80, 0.84)]:
                arrow((x + w * start, inner_y + h * 0.14), (x + w * end, inner_y + h * 0.14), edge, linewidth=0.55, mutation_scale=5)

        ax.text(0.115, 0.955, "Protein sequence", fontsize=7.3, fontweight="bold", ha="left", va="center")
        ax.text(0.175, 0.925, "...", fontsize=8.5, ha="center", va="center")
        token_track(0.210, 0.910, "MASSCAVQKLEP", 0.039, 0.030)
        ax.text(0.690, 0.925, "...", fontsize=8.5, ha="center", va="center")
        arrow((0.455, 0.910), (0.455, 0.890), palette["black"], linewidth=0.75, mutation_scale=8)

        panel(0.060, 0.700, 0.880, 0.190, "a", "Residue representation layer")
        inset(0.090, 0.735, 0.270, 0.115, "Frozen ESM2-t33 embeddings", palette["blue"])
        matrix(0.113, 0.757, 6, 14, 0.220, 0.055, palette["blue"], "esm")
        vector_strip(0.115, 0.742, 14, 0.216, 0.008, palette["blue"], palette["pale_blue"])
        ax.text(0.315, 0.829, "pLM", fontsize=5.5, color=palette["blue"], ha="center", va="center", fontweight="bold")

        inset(0.390, 0.735, 0.235, 0.115, "Amino-acid identity", palette["green"])
        matrix(0.415, 0.760, 5, 12, 0.180, 0.050, palette["green"], "onehot")
        vector_strip(0.415, 0.742, 12, 0.180, 0.008, palette["green"], palette["pale_green"])

        inset(0.655, 0.735, 0.215, 0.115, "Relative position", palette["vermillion"])
        matrix(0.678, 0.760, 4, 12, 0.170, 0.050, palette["vermillion"], "position")
        vector_strip(0.680, 0.742, 12, 0.166, 0.008, palette["vermillion"], palette["pale_orange"])

        box(0.210, 0.712, 0.500, 0.026, "Concatenate residue-wise features", "#FFFFFF", palette["black"], 5.7, linewidth=0.65, radius=0.004)
        vector_strip(0.305, 0.705, 20, 0.300, 0.006, palette["gray"], "#FFFFFF")
        box(0.758, 0.705, 0.070, 0.036, "x_i", "#FFFFFF", palette["black"], 6.0, "bold")
        arrow((0.710, 0.725), (0.758, 0.723), palette["gray"], linewidth=0.50, mutation_scale=6)
        arrow((0.224, 0.735), (0.430, 0.738), palette["blue"], curve=-0.08, linewidth=0.40, mutation_scale=5)
        arrow((0.505, 0.735), (0.485, 0.738), palette["green"], curve=0.03, linewidth=0.40, mutation_scale=5)
        arrow((0.765, 0.735), (0.575, 0.738), palette["vermillion"], curve=0.08, linewidth=0.40, mutation_scale=5)

        arrow((0.500, 0.700), (0.500, 0.670), palette["black"], linewidth=0.75, mutation_scale=8)

        panel(0.060, 0.488, 0.880, 0.182, "b", "Warm-start shared temporal encoder")
        box(0.095, 0.575, 0.082, 0.052, "LayerNorm\n+ linear", palette["light_gray"], palette["gray"], 5.4)
        vector_strip(0.099, 0.538, 10, 0.074, 0.017, palette["gray"], "#FFFFFF")
        ax.text(0.103, 0.521, "h_i^0", fontsize=5.4, color=palette["gray"], ha="left", va="center")
        arrow((0.177, 0.601), (0.235, 0.564), palette["gray"], linewidth=0.65, mutation_scale=7)

        ax.text(0.238, 0.628, "Multi-kernel residual TCN stack", fontsize=6.0, ha="left", va="center", fontweight="bold")
        dilations = [1, 2, 4, 8]
        for index, dilation in enumerate(dilations):
            conv_block(0.245 + index * 0.080, 0.520, 0.064, 0.088, dilation)
            if index < len(dilations) - 1:
                arrow((0.309 + index * 0.080, 0.564), (0.325 + index * 0.080, 0.564), palette["gray"], linewidth=0.45, mutation_scale=5)
        box(0.625, 0.552, 0.110, 0.055, "sequence state\nh_i", "#FFFFFF", palette["black"], 5.7, "bold")
        vector_strip(0.635, 0.520, 12, 0.090, 0.013, palette["black"], palette["pale_blue"])
        arrow((0.565, 0.564), (0.625, 0.579), palette["gray"], linewidth=0.60, mutation_scale=7)
        box(0.750, 0.552, 0.118, 0.055, "P4.6 checkpoint\nwarm-start", palette["pale_blue"], palette["blue"], 5.3)
        arrow((0.807, 0.552), (0.530, 0.520), palette["blue"], curve=-0.18, linewidth=0.50, mutation_scale=6)
        ax.text(0.103, 0.504, "Input projection and TCN weights are frozen in P4.8.", fontsize=5.3, color=palette["gray"], ha="left", va="center")

        arrow((0.500, 0.488), (0.500, 0.458), palette["black"], linewidth=0.75, mutation_scale=8)

        panel(0.060, 0.278, 0.880, 0.180, "c", "Region-aware low-rank adapter mixture")
        box(0.095, 0.360, 0.082, 0.052, "Shared\nh_i", "#FFFFFF", palette["black"], 5.8, "bold")
        adapter_specs = [
            ("SDR", 0.235, 0.388, palette["blue"], palette["pale_blue"]),
            ("LDR", 0.235, 0.338, palette["green"], palette["pale_green"]),
            ("Terminal-IDR", 0.495, 0.388, palette["vermillion"], palette["pale_orange"]),
            ("Internal-IDR", 0.495, 0.338, palette["purple"], palette["pale_purple"]),
        ]
        for label, x, y, edge, face in adapter_specs:
            adapter_block(x, y, 0.205, 0.040, label, edge, face)
            arrow((0.177, 0.386), (x, y + 0.020), edge, curve=0.05 if x < 0.30 else -0.05, linewidth=0.45, mutation_scale=5)
        box(0.095, 0.298, 0.082, 0.044, "Gate head\nsoftmax", "#FFFFFF", palette["purple"], 5.0, "bold", linestyle="--")
        arrow((0.136, 0.360), (0.136, 0.342), palette["purple"], linewidth=0.45, mutation_scale=5, linestyle="--")
        ax.text(0.730, 0.420, "MoE weights w_i", fontsize=5.5, ha="left", va="center")
        weights = [0.30, 0.22, 0.19, 0.29]
        weight_colors = [palette["blue"], palette["green"], palette["vermillion"], palette["purple"]]
        for index, (weight, color) in enumerate(zip(weights, weight_colors)):
            ax.add_patch(Rectangle((0.725, 0.383 - index * 0.018), 0.065 * weight / max(weights), 0.010, facecolor=color, edgecolor=color, linewidth=0.3))
        box(0.825, 0.342, 0.075, 0.055, "sum\nw_r e_r", "#FFFFFF", palette["black"], 5.4, "bold")
        for _, x, y, edge, _ in adapter_specs:
            arrow((x + 0.205, y + 0.020), (0.825, 0.370), edge, curve=0.04 if y > 0.36 else -0.04, linewidth=0.45, mutation_scale=5)
        arrow((0.177, 0.320), (0.825, 0.352), palette["purple"], curve=-0.12, linewidth=0.45, mutation_scale=5, linestyle="--")
        ax.text(0.238, 0.292, "Each expert uses a residual low-rank adapter: Down 128->32, GELU, Up 32->128, and a residue logit head.", fontsize=5.0, color=palette["gray"], ha="left", va="center")

        arrow((0.500, 0.278), (0.500, 0.250), palette["black"], linewidth=0.75, mutation_scale=8)

        panel(0.060, 0.060, 0.880, 0.190, "d", "Output layer, calibration and uncertainty")
        x0 = 0.125
        xs = [x0 + i * 0.030 for i in range(20)]
        token_track(0.126, 0.203, "MASSCAVQKLEP", 0.030, 0.017)
        score_tracks = [
            [0.18, 0.22, 0.30, 0.43, 0.61, 0.72, 0.67, 0.52, 0.38, 0.29, 0.24, 0.33, 0.55, 0.78, 0.83, 0.69, 0.47, 0.31, 0.22, 0.18],
            [0.15, 0.21, 0.28, 0.40, 0.58, 0.70, 0.65, 0.55, 0.39, 0.30, 0.25, 0.36, 0.58, 0.76, 0.80, 0.70, 0.50, 0.34, 0.25, 0.20],
            [0.20, 0.23, 0.31, 0.44, 0.63, 0.73, 0.68, 0.54, 0.36, 0.28, 0.23, 0.34, 0.53, 0.74, 0.82, 0.67, 0.45, 0.30, 0.23, 0.18],
        ]
        labels = ["seed 1", "seed 2", "seed 3"]
        for row, scores in enumerate(score_tracks):
            y = 0.176 - row * 0.021
            ax.plot(xs, [y + score * 0.014 for score in scores], color="#56B4E9", linewidth=0.60)
            ax.text(0.088, y + 0.007, labels[row], fontsize=4.8, color=palette["gray"], ha="left", va="center")
        box(0.520, 0.158, 0.080, 0.040, "Average\n3 seeds", palette["pale_blue"], palette["blue"], 5.1)
        box(0.625, 0.158, 0.080, 0.040, "Platt\ncalibration", palette["pale_yellow"], "#8A6D00", 5.1)
        arrow((0.472, 0.174), (0.520, 0.178), palette["blue"], linewidth=0.45, mutation_scale=5)
        arrow((0.600, 0.178), (0.625, 0.178), "#8A6D00", linewidth=0.45, mutation_scale=5)
        prob = [sum(values) / len(values) for values in zip(*score_tracks)]
        ax.plot(xs, [0.102 + value * 0.033 for value in prob], color=palette["black"], linewidth=0.95)
        ax.axhline(0.102 + 0.50 * 0.033, xmin=0.118, xmax=0.706, color=palette["gray"], linewidth=0.40, linestyle="--")
        ax.text(0.088, 0.120, "p(IDR)", fontsize=5.2, ha="left", va="center", fontweight="bold")
        for x, value in zip(xs, prob):
            face = palette["vermillion"] if value >= 0.50 else "#FFFFFF"
            ax.add_patch(Rectangle((x - 0.009, 0.082), 0.018, 0.013, facecolor=face, edgecolor=palette["vermillion"], linewidth=0.35))
        ax.text(0.088, 0.089, "binary", fontsize=5.0, ha="left", va="center")
        entropy = [min(1.0, 4.0 * value * (1.0 - value)) for value in prob]
        for x, value in zip(xs, entropy):
            ax.add_patch(Rectangle((x - 0.007, 0.064), 0.014, value * 0.018, facecolor="#E8B5CE", edgecolor=palette["purple"], linewidth=0.18))
        ax.text(0.088, 0.066, "entropy", fontsize=5.0, ha="left", va="center")
        arrow((0.705, 0.178), (0.735, 0.140), palette["black"], linewidth=0.45, mutation_scale=5)
        ax.text(0.745, 0.140, "Calibrated residue probability,\nbinary disorder calls and\nentropy uncertainty", fontsize=5.2, ha="left", va="center", linespacing=1.05)
        ax.text(0.745, 0.087, "Calibration parameters are fitted\nonly on DM1229 validation.", fontsize=4.9, color=palette["gray"], ha="left", va="center", linespacing=1.05)

        fig.savefig(path_pdf, dpi=600, metadata={"Title": "RegionAdapterMoETCN method overview"})
        fig.savefig(path_eps, dpi=600)
        fig.savefig(path_svg, dpi=600)
        fig.savefig(path_png, dpi=600)
        plt.close(fig)

    from PIL import Image

    with Image.open(path_png) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(path_png, dpi=(600, 600))

    manifest = {
        "figure": "P5.8 RegionAdapterMoETCN method overview",
        "audience_medium": "Bioinformatics manuscript, static full-width method figure",
        "figure_type": "multi-panel method architecture schematic; no quantitative data encoding",
        "target_size_mm": {"width": width_mm, "height": height_mm},
        "source_inputs": [
            "scripts/assemble_p5_8_fusionencoder_style_assets.py",
            "manuscript/latex/bioinformatics/main.tex",
            "models/sequence_models.py",
            "scripts/train_sequence_disorder_model.py",
        ],
        "transformations": [
            "manual architecture-figure layout from the P4.8 model description and local reference method figures",
            "no numeric data filtering, smoothing, normalization, or imputation",
        ],
        "accessibility": [
            "all module roles are labeled directly",
            "color is redundant with text labels, panel separation and dashed line styles",
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
