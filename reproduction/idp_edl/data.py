"""FASTA parsing and the IDP-EDL evaluation data contract.

The author files contain sequence headers with a dataset suffix and label
headers with an SDR/LDR suffix.  The two files are aligned by record order,
while the first header token is checked as an additional identity guard.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DATASET_ORDER = ("mxd494", "sl329", "disorder723")
DATASET_ALIASES = {
    "mxd494": "mxd494",
    "mxd": "mxd494",
    "sl329": "sl329",
    "sl": "sl329",
    "disorder723": "disorder723",
    "disorder": "disorder723",
}


@dataclass(frozen=True)
class FastaRecord:
    """A FASTA record retaining the exact header and sequence text."""

    header: str
    sequence: str

    @property
    def identifier(self) -> str:
        tokens = self.header.split()
        if not tokens:
            raise ValueError("FASTA header is empty")
        return tokens[0]

    @property
    def category(self) -> str:
        tokens = self.header.split()
        return tokens[-1].lower() if len(tokens) > 1 else ""


@dataclass(frozen=True)
class LabeledExample:
    """One author sample sequence and its residue labels."""

    dataset: str
    identifier: str
    sequence_header: str
    label_header: str
    sequence: str
    labels: str
    label_category: str


@dataclass(frozen=True)
class TruncationStats:
    """Residue accounting under the author's tokenizer convention."""

    original_length: int
    retained_length: int
    truncated_residues: int
    max_length: int
    eos_tokens: int


def canonical_dataset(name: str) -> str:
    value = name.strip().lower().replace("_test", "")
    if value == "all":
        return "all"
    try:
        return DATASET_ALIASES[value]
    except KeyError:
        choices = ", ".join(("all",) + DATASET_ORDER)
        raise ValueError("unknown dataset {!r}; choose {}".format(name, choices))


def read_fasta(path: Path) -> List[FastaRecord]:
    """Read a plain FASTA file without changing headers or residue order."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("FASTA file does not exist: {}".format(path))

    records = []
    header = None
    sequence_parts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(sequence_parts).replace(" ", "")
                    if not sequence:
                        raise ValueError("empty sequence for {} at {}:{}".format(header, path, line_number))
                    records.append(FastaRecord(header, sequence))
                header = line[1:].strip()
                if not header:
                    raise ValueError("empty FASTA header at {}:{}".format(path, line_number))
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError("sequence appears before the first header at {}:{}".format(path, line_number))
                sequence_parts.append("".join(line.split()))

    if header is not None:
        sequence = "".join(sequence_parts).replace(" ", "")
        if not sequence:
            raise ValueError("empty sequence for {} at end of {}".format(header, path))
        records.append(FastaRecord(header, sequence))
    if not records:
        raise ValueError("FASTA file contains no records: {}".format(path))
    return records


def _validate_labels(labels: str, header: str) -> None:
    invalid = sorted(set(labels).difference(set("012")))
    if invalid:
        raise ValueError("unsupported label characters {} in {}".format(invalid, header))


def parse_author_samples(
    sequence_path: Path,
    label_path: Path,
    datasets: Optional[Iterable[str]] = None,
) -> List[LabeledExample]:
    """Parse and align the author's sample FASTA pair.

    The official files reuse identifiers across different benchmark suffixes,
    so alignment is deliberately by record order with an ID and length check.
    This matches the author's files and avoids an ambiguous global ID join.
    """

    sequence_records = read_fasta(Path(sequence_path))
    label_records = read_fasta(Path(label_path))
    if len(sequence_records) != len(label_records):
        raise ValueError(
            "sequence/label record count mismatch: {} != {}".format(
                len(sequence_records), len(label_records)
            )
        )

    selected = None
    if datasets is not None:
        selected = set()
        for name in datasets:
            canonical = canonical_dataset(name)
            if canonical == "all":
                selected.update(DATASET_ORDER)
            else:
                selected.add(canonical)

    examples = []
    for index, (sequence_record, label_record) in enumerate(zip(sequence_records, label_records)):
        if sequence_record.identifier != label_record.identifier:
            raise ValueError(
                "sequence/label ID mismatch at record {}: {} != {}".format(
                    index, sequence_record.header, label_record.header
                )
            )
        if len(sequence_record.sequence) != len(label_record.sequence):
            raise ValueError(
                "sequence/label length mismatch for {}: {} != {}".format(
                    sequence_record.identifier,
                    len(sequence_record.sequence),
                    len(label_record.sequence),
                )
            )
        _validate_labels(label_record.sequence, label_record.header)
        dataset = sequence_record.category
        if selected is not None and dataset not in selected:
            continue
        examples.append(
            LabeledExample(
                dataset=dataset,
                identifier=sequence_record.identifier,
                sequence_header=sequence_record.header,
                label_header=label_record.header,
                sequence=sequence_record.sequence,
                labels=label_record.sequence,
                label_category=label_record.category,
            )
        )
    return examples


def parse_sequence_fasta(path: Path, dataset: str = "input") -> List[LabeledExample]:
    """Read a sequence-only FASTA for prediction without synthetic labels."""

    records = read_fasta(Path(path))
    return [
        LabeledExample(
            dataset=dataset,
            identifier=record.identifier,
            sequence_header=record.header,
            label_header="",
            sequence=record.sequence,
            labels="",
            label_category="",
        )
        for record in records
    ]


def single_sequence(sequence: str, header: str = "smoke_1") -> List[LabeledExample]:
    value = "".join(sequence.split())
    if not value:
        raise ValueError("sequence input is empty")
    return [
        LabeledExample(
            dataset="smoke",
            identifier=header.split()[0],
            sequence_header=header,
            label_header="",
            sequence=value,
            labels="",
            label_category="",
        )
    ]


def normalize_for_prott5(sequence: str) -> str:
    """Match the official replacement of unsupported amino-acid symbols."""

    return "".join("X" if residue in "OBUZ" else residue for residue in sequence)


def truncation_stats(sequence_length: int, max_length: int = 1024) -> TruncationStats:
    """Return author-compatible accounting: max_length includes one EOS token."""

    if max_length < 2:
        raise ValueError("max_length must be at least 2 to leave one EOS position")
    retained = min(sequence_length, max_length - 1)
    return TruncationStats(
        original_length=sequence_length,
        retained_length=retained,
        truncated_residues=sequence_length - retained,
        max_length=max_length,
        eos_tokens=1,
    )


def truncate_labels(labels: str, sequence_length: int, max_length: int = 1024) -> Tuple[str, TruncationStats]:
    if len(labels) != sequence_length:
        raise ValueError("label length does not match sequence length")
    stats = truncation_stats(sequence_length, max_length)
    return labels[: stats.retained_length], stats


def default_sample_paths(root: Optional[Path] = None) -> Tuple[Path, Path]:
    if root is None:
        root = Path(__file__).resolve().parents[2]
    data_dir = Path(root) / "external" / "IDP-EDL" / "data"
    return data_dir / "sample_sequences.fasta", data_dir / "sample_labels.fasta"
