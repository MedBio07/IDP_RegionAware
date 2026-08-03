# DM3000 NR25 Training Sets

These files were generated from `data/DM3000_Train.fasta` by searching each test set against DM3000 with MMseqs2 and removing any DM3000 training sequence with an alignment sequence identity (`pident`) greater than 25% to at least one sequence in the corresponding test set.

MMseqs2 command core: `easy-search <test_sequences> <DM3000_sequences> <hits.m8> <tmp> --min-seq-id 0.25 -s 7.5 --max-seqs 100000`.

Output training FASTA files preserve the original three-line labeled FASTA format: header, sequence, label list.
