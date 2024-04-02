import numpy as np
import logging
import scbamtools.util as util

logger = logging.getLogger("random")


def wrap(seq, n=80):
    buffer = []
    while seq:
        l = min(n, len(seq))
        buffer.append(seq[:l])
        seq = seq[l:]

    return "\n".join(buffer)


class Genome:
    def __init__(self, seqs):
        self.seqs = seqs
        self.chrom_sizes = {}
        for chrom, seq in self.seqs.items():
            self.chrom_sizes[chrom] = len(seq)

    def get_oriented(self, chrom, start, end, strand):
        seq = self.seqs[chrom][start:end]
        if strand == "-":
            seq = util.rev_comp(seq)

        return seq

    def store_fa(self, fname):
        with open(fname, "w") as fa:
            for chrom, seq in sorted(self.seqs.items()):
                wrapped = wrap(seq)
                fa.write(f">{chrom}\n{wrapped}\n\n")


def make_genome(chrom_sizes):
    genome = {}
    for chrom, size in sorted(chrom_sizes.items()):
        seq = "".join(np.random.choice(np.array(list("ACGT")), size=size, replace=True))
        genome[chrom] = seq

    return Genome(genome)


logging.basicConfig(level=logging.INFO)
np.random.seed(13770815)

logger.info("generating genome")
chrom_sizes = {}
for line in open("test_data/simple_chrom_sizes.csv"):
    chrom, size = line.rstrip().split("\t")
    chrom_sizes[chrom] = int(size)

genome = make_genome(chrom_sizes)
genome.store_fa("test_data/simple_genome.fa")

logger.info("sampling cDNA reads")


def sample_reads(mrna, l=90, n=200):
    for i in np.random.randint(0, len(mrna) - l + 1, size=n):
        yield mrna[i : i + l]


from scbamtools.gene_model import Transcript

import gzip

with open("test_data/simple_mrnas.fa", "w") as mrna:
    with gzip.open("test_data/simple_reads.fastq.gz", mode="wt") as fq:
        for tx in Transcript.from_GTF("test_data/simple_annotation.gtf", genome=genome):
            seq = tx.spliced_sequence
            mrna.write(f">{tx.transcript_id}\n{seq}\n")
            for i, read in enumerate(sample_reads(seq)):
                qual = "I" * len(read)
                fq.write(f"@read_{tx.transcript_id}_{i}\n{read}\n+\n{qual}\n")
