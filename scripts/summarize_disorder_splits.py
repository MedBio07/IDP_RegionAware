#!/usr/bin/env python3
"""Generate the default DM3000 disorder split and region summaries."""

from __future__ import annotations

from pathlib import Path

from annotate_disorder_regions import default_specs, run


def main() -> None:
    run(
        dataset_specs=default_specs(),
        protein_out=Path("results/region_annotations/protein_region_annotations.tsv"),
        segments_out=Path("results/region_annotations/disorder_segments.tsv"),
        summary_out=Path("results/dataset_region_summary.tsv"),
        test_summary_out=Path("results/testset_region_summary.tsv"),
    )


if __name__ == "__main__":
    main()
