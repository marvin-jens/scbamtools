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


def random_sequences(l=12, n=1, alphabet=list("ACGT")):
    for i in range(n):
        yield "".join(np.random.choice(alphabet, size=l, replace=True))


cell_barcodes = list(random_sequences(l=12, n=5))
UMIs = list(random_sequences(l=8, n=150))


def make_genome(chrom_sizes):
    genome = {}
    for chrom, size in sorted(chrom_sizes.items()):
        seq = next(random_sequences(l=size, n=1))
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


def sample_barcodes(i):
    return cell_barcodes[i % len(cell_barcodes)], UMIs[i % len(UMIs)]


def sample_quality(l=50, l_good=80, phred_base=33, min_qual=2, max_qual=40):
    x = np.arange(l)
    m = np.where(x < l_good, max_qual - 2, min_qual)
    M = np.where(x < l_good, max_qual, max_qual - 10)
    Q = np.random.randint(m, high=M + 1, size=l)

    return "".join([chr(q + phred_base) for q in Q])


from scbamtools.gene_model import Transcript

import gzip

with open("test_data/simple_mrnas.fa", "w") as mrna:
    with gzip.open("test_data/simple.reads1.fastq.gz", mode="wt") as fq1:
        with gzip.open("test_data/simple.reads2.fastq.gz", mode="wt") as fq2:
            for tx in Transcript.from_GTF(
                "test_data/simple_annotation.gtf", genome=genome
            ):
                seq = tx.spliced_sequence + "A" * 90
                mrna.write(f">{tx.transcript_id}\n{seq}\n")
                for i, read in enumerate(sample_reads(seq)):
                    qual = sample_quality(len(read))
                    fq2.write(f"@read_{tx.transcript_id}_{i}\n{read}\n+\n{qual}\n")

                    bc, umi = sample_barcodes(i)
                    seq1 = bc + umi + "TT"
                    qual = sample_quality(len(seq1), l_good=10)  # "I" * len(seq1)
                    fq1.write(f"@read_{tx.transcript_id}_{i}\n{seq1}\n+\n{qual}\n")

                seq = tx.unspliced_sequence + "A" * 90
                mrna.write(f">{tx.transcript_id}\n{seq}\n")
                for i, read in enumerate(sample_reads(seq)):
                    qual = sample_quality(len(read))
                    fq2.write(f"@read_{tx.transcript_id}_{i}\n{read}\n+\n{qual}\n")

                    bc, umi = sample_barcodes(i)
                    seq1 = bc + umi + "TT"
                    qual = sample_quality(len(seq1), l_good=10)  # "I" * len(seq1)
                    fq1.write(f"@read_{tx.transcript_id}_{i}\n{seq1}\n+\n{qual}\n")
