from collections import namedtuple
import numpy as np
import re
from bisect import bisect_left, bisect_right

Gff = namedtuple(
    "gff",
    ["chrom", "source", "type", "start", "end", "score", "strand", "frame", "attr_str"],
)

Locus = namedtuple("Locus", ["name", "chrom", "start", "end", "strand", "type"])


def attr_to_dict(attr_str: str) -> dict:
    """
    Helper for reading the GTF. Parsers the attr_str found in the GTF
    attribute column (col 9). Returns a dictionary from the key, value
    pairs.
    """
    d = {}
    for match in re.finditer(r"(\w+) \"(\S+)\";", attr_str):
        key, value = match.groups()
        d[key] = value

    if not "gene_name" in d:
        d["gene_name"] = d.get("gene_id", d.get("transcript_id", "na"))

    return d


def GFF_importer(path):
    for line in open(path, "rt"):
        if line.startswith("#"):
            continue

        line = line.rstrip()
        if not line:
            continue

        cols = line.split("\t")
        cols[3] = int(cols[3])
        cols[4] = int(cols[4])
        # cols[-1] = attr_to_dict(cols[-1])

        yield Gff(*cols)


class ExonChain(object):
    """
    Implements linked blocks of genomic coordinates (with orientation).
    Features splicing from arbitrary tracks (including sequence) and intersection with
    simple start/end blocks on the genome. The latter returns 3 new ExonChains. As an
    example consider intersection with genomic CDS start and end-coordinates. This will
    return 5'UTR, CDS and 3'UTR as individual ExonChains.
    This class is deliberately kept as light-weight as possible. It has no means of
    generating names/annotation. All of that is handled by the "Transcript" subclass.
    """

    def __init__(self, chrom, strand, exon_starts, exon_ends, genome, name=""):
        self.genome = genome
        self.chrom = chrom
        self.strand = strand
        self.sense = strand  # deprecated
        self.exon_starts = np.array(sorted(exon_starts))
        self.exon_ends = np.array(sorted(exon_ends))
        self.name = name
        self._setup()

    def _setup(self):
        self.start = self.exon_starts[0]
        self.end = self.exon_ends[-1]

        self.exon_count = len(self.exon_starts)
        self.exon_bounds = np.array([self.exon_starts, self.exon_ends]).transpose()
        self.intron_bounds = []
        if self.exon_count > 1:
            self.intron_bounds = np.array(
                [self.exon_ends[:-1], self.exon_starts[1:]]
            ).transpose()

        self.exon_lengths = self.exon_ends - self.exon_starts
        self.exon_txstarts = np.array(
            [
                0,
            ]
            + list(self.exon_lengths.cumsum())
        )

        self.spliced_length = max(self.exon_txstarts[-1], 0)
        self.unspliced_length = self.end - self.start

        if self.strand == "-":
            self.dir = -1
            self.ofs = self.spliced_length
        else:
            self.dir = +1
            self.ofs = 0

        self.tx_start, self.tx_end = [self.start, self.end][:: self.dir]
        if not self.name:
            # if the chain gets altered subsequently, don't loose the name
            self.name = "ExonChain_%s:%d-%d%s__%s__%s" % (
                self.chrom,
                self.start,
                self.end,
                self.strand,
                ",".join([str(s) for s in self.exon_starts]),
                ",".join([str(e) for e in self.exon_ends]),
            )

    def upstream(self, L=100):
        if self.strand == "+":
            exon_starts = [self.start - L]
            exon_ends = [self.start]
        else:
            exon_starts = [self.end]
            exon_ends = [self.end + L]

        return ExonChain(
            self.chrom,
            self.strand,
            exon_starts,
            exon_ends,
            self.genome,
            f"{self.name}_upstream_{L}",
        )

    def downstream(self, L=100):
        if self.strand == "+":
            exon_starts = [self.end]
            exon_ends = [self.end + L]
        else:
            exon_starts = [self.start - L]
            exon_ends = [self.start]

        return ExonChain(
            self.chrom,
            self.strand,
            exon_starts,
            exon_ends,
            self.genome,
            f"{self.name}_downstream_{L}",
        )

    @property
    def intron_chain(self):
        if self.exon_count > 1:
            return ExonChain(
                self.chrom,
                self.strand,
                self.exon_ends[:-1],
                self.exon_starts[1:],
                genome=self.genome,
            )

    def map_to_spliced(self, pos, truncate=False):
        if self.strand == "+":
            n = min(bisect_right(self.exon_starts, pos), self.exon_count) - 1
            if not (self.exon_starts[n] <= pos <= self.exon_ends[n]):
                if not truncate:
                    raise ValueError("%d does not lie within any exon bounds" % pos)
                else:
                    pos = self.exon_ends[n]

            return self.ofs + self.dir * (
                pos - self.exon_starts[n] + self.exon_txstarts[n]
            )
        else:
            n = min(max(0, bisect_left(self.exon_ends, pos)), self.exon_count - 1)
            if not (self.exon_starts[n] <= pos <= self.exon_ends[n]):
                if not truncate:
                    raise ValueError("%d does not lie within any exon bounds" % pos)
                else:
                    pos = self.exon_starts[n]

            return (
                self.ofs
                + self.dir * (pos - self.exon_starts[n] + self.exon_txstarts[n])
                - 1
            )

    def map_block_from_spliced(self, start, end):
        x, y = self.map_from_spliced(start), self.map_from_spliced(end)
        if y > x:
            # semantics change here because start is included, end is excluded
            # in C indexing.
            return x, y
        else:
            return y, x
        # return min(x,y),max(x,y)

    def map_block_to_spliced(self, start, end, truncate=False):
        x, y = self.map_to_spliced(start, truncate), self.map_to_spliced(end, truncate)
        if y > x:
            # semantics change here because start is included, end is excluded
            # in C indexing.
            return x, y
        else:
            return y, x

    def map_to_exon(self, tx_pos):
        pos = self.ofs + self.dir * tx_pos
        n = max(0, bisect_left(self.exon_txstarts, pos) - 1)
        return n

    def map_from_spliced(self, pos):
        pos = self.ofs + self.dir * pos
        if self.strand == "+":
            n = min(bisect_right(self.exon_txstarts, pos), self.exon_count) - 1
            if n == self.exon_count or n < 0:
                raise ValueError("{0} out of transcript bounds".format(pos))

            return self.exon_starts[n] + pos - self.exon_txstarts[n]
        else:
            n = max(0, bisect_left(self.exon_txstarts, pos) - 1)
            if n == self.exon_count or n < 0:
                raise ValueError("{0} out of transcript bounds".format(pos))

            return self.exon_starts[n] + pos - self.exon_txstarts[n] - 1

    def splice(self, track, join=lambda l: "".join(l), get="get_oriented", **kwargs):
        get_func = getattr(track, get)
        return join(
            [
                get_func(self.chrom, start, end, self.strand, **kwargs)
                for start, end in self.exon_bounds
            ][:: self.dir]
        )

    @property
    def unspliced_sequence(self):
        return self.genome.get_oriented(self.chrom, self.start, self.end, self.strand)

    @property
    def spliced_sequence(self):
        return self.splice(self.genome)

    @property
    def splice_sites(self):
        """returns tuples of (5'ss, 3'ss)"""
        for bounds in self.intron_bounds[:: self.dir]:
            yield bounds[:: self.dir]

    @property
    def exons(self):
        for i, (start, end) in enumerate(self.exon_bounds[:: self.dir]):
            yield Locus(
                "%s/exon.%02d" % (self.name, i + 1),
                self.chrom,
                self.strand,
                start,
                end,
                "exon",
            )

    @property
    def introns(self):
        for i, (start, end) in enumerate(self.intron_bounds[:: self.dir]):
            yield Locus(
                "%s/intron.%02d" % (self.name, i + 1),
                self.chrom,
                self.strand,
                start,
                end,
                "intron",
            )

    @property
    def introns_as_chains(self):
        for i, (start, end) in enumerate(self.intron_bounds[:: self.dir]):
            yield ExonChain(
                self.chrom,
                self.strand,
                [
                    start,
                ],
                [
                    end,
                ],
                name="%s/intron.%02d" % (self.name, i + 1),
                genome=self.genome,
            )

    def cut(self, start, end, expand=False):
        names = ["before", "cut", "after"]
        return self.intersect(names, start, end, expand=expand)

    def intersect(self, names, start, end, expand=False):
        # TODO: Clean this up
        # print "INTERSECT",names,start,end
        start, end = min(start, end), max(start, end)
        if not expand:
            start = max(start, self.start)
            end = min(end, self.end)

        first = bisect_right(self.exon_starts, start) - 1
        last = bisect_right(self.exon_starts, end) - 1

        f2 = bisect_right(self.exon_ends, start)
        l2 = bisect_right(self.exon_ends, end)

        # print first,f2
        # print last,l2

        before_starts = list(self.exon_starts[: f2 + 1])
        before_ends = list(self.exon_ends[: f2 + 1])

        if not before_starts and not before_ends:
            before_starts = [0]
            before_ends = [0]

        chain_starts = list(self.exon_starts[f2 : last + 1])
        chain_ends = list(self.exon_ends[f2 : last + 1])

        if not chain_starts and not chain_ends:
            chain_starts = [0]
            chain_ends = [0]

        # if not chain_starts:
        # print first,last,start,end,self.tx_start,self.tx_end

        after_starts = list(self.exon_starts[last:])
        after_ends = list(self.exon_ends[last:])

        if not after_starts and not after_ends:
            after_starts = [0]
            after_ends = [0]

        if first == f2:
            chain_starts[0] = max(start, chain_starts[0])
            before_ends[-1] = min(start, before_ends[-1])
        else:
            # print("start falls between exon bounds?")
            # truncate chain, breakpoint between two exons
            before_ends[-1] = before_starts[-1]
            if expand:
                # expand the exon-bound to incorporate the start
                chain_starts[0] = start

        if last == l2:
            chain_ends[-1] = min(end, chain_ends[-1])
            after_starts[0] = max(end, after_starts[0])
        else:
            # print("end falls between exon bounds?")
            # truncate chain, breakpoint between two exons
            after_starts[0] = after_ends[0]
            if expand:
                # expand the exon-bound to incorporate the end
                chain_ends[-1] = end

        def remove_empty(starts, ends):
            keep_starts = []
            keep_ends = []
            for s, e in zip(starts, ends):
                if s != e:
                    keep_starts.append(s)
                    keep_ends.append(e)

            return keep_starts, keep_ends

        before_starts, before_ends = remove_empty(before_starts, before_ends)
        chain_starts, chain_ends = remove_empty(chain_starts, chain_ends)
        after_starts, after_ends = remove_empty(after_starts, after_ends)

        if chain_starts:
            chain = ExonChain(
                self.chrom, self.strand, chain_starts, chain_ends, genome=self.genome
            )
        else:
            chain = self.EmptyChain()

        if before_starts:
            before = ExonChain(
                self.chrom, self.strand, before_starts, before_ends, genome=self.genome
            )
        else:
            before = self.EmptyChain()

        if after_starts:
            after = ExonChain(
                self.chrom, self.strand, after_starts, after_ends, genome=self.genome
            )
        else:
            after = self.EmptyChain()

        if self.strand == "+":
            return before, chain, after
        else:
            return after, chain, before

    def cut(self, start, end, expand=False):
        before, chain, after = self.intersect(
            ["before", "cut", "after"], start, end, expand=expand
        )
        return chain

    @property
    def key(self):
        return (self.chrom, self.strand, tuple(self.exon_starts), tuple(self.exon_ends))

    @property
    def key_str(self):
        return "{self.chrom}:{es}-{ee}{self.strand}".format(
            self=self,
            es=",".join([str(x) for x in self.exon_starts]),
            ee=",".join([str(x) for x in self.exon_ends]),
        )

    def __add__(self, other):
        """
        concatenates two non-overlapping exon-chains, returning a new ExonChain object
        """

        # ensure A precedes B in chromosome
        if self.start < other.start:
            A, B = self, other
        else:
            A, B = other, self

        if not A:
            return B
        if not B:
            return A
        # print "# concatenating exon chains",A,B
        assert A.chrom == B.chrom
        assert A.strand == B.strand
        assert A.genome == B.genome
        assert A.end <= B.start  # support only non-overlapping chains for now!

        # these must now stay in order and can not overlap!
        exon_starts = np.concatenate((A.exon_starts, B.exon_starts))
        exon_ends = np.concatenate((A.exon_ends, B.exon_ends))

        return ExonChain(A.chrom, A.strand, exon_starts, exon_ends, genome=A.genome)

    # add some python magic to make things smooth
    def __str__(self):
        exonlist = ",".join(map(str, self.exon_bounds))
        return "{self.chrom}:{self.start}-{self.end}{self.strand} spliced_length={self.spliced_length} exons: {exonlist}".format(
            self=self, exonlist=exonlist
        )

    def bed12_format(self, color="255,0,0"):
        block_sizes = [str(e - s) for s, e in self.exon_bounds]
        block_starts = [str(s - self.start) for s in self.exon_starts]

        if getattr(self, "CDS", None):
            cds_start = self.CDS.start
            cds_end = self.CDS.end
        else:
            cds_start = self.end
            cds_end = self.end

        cols = [
            self.chrom,
            self.start,
            self.end,
            self.name,
            getattr(self, "score", 0),
            self.strand,
            cds_start,
            cds_end,
            color,
            self.exon_count,
            ",".join(block_sizes),
            ",".join(block_starts),
        ]
        return "\t".join([str(c) for c in cols])

    def ucsc_format(self):
        exonstarts = ",".join([str(s) for s in self.exon_starts])
        exonends = ",".join([str(e) for e in self.exon_ends])
        # exonframes = ",".join([str(f) for f in self.exon_frames])
        # TODO: fix exon-frames
        exonframes = ",".join(["-1" for e in self.exon_ends])

        if getattr(self, "CDS", None):
            cds_start = self.CDS.start
            cds_end = self.CDS.end
        else:
            cds_start = self.end
            cds_end = self.end

        out = (
            -1,
            self.name,
            self.chrom,
            self.strand,
            self.start,
            self.end,
            cds_start,
            cds_end,
            self.exon_count,
            exonstarts,
            exonends,
            getattr(self, "score", 0),
            getattr(self, "gene_id", self.name),
            "unk",
            "unk",
            exonframes,
        )
        return "\t".join([str(o) for o in out])

    def __len__(self):
        """
        zero-length ExonChains will be False in truth value testing.
        so stuff like: "if tx.UTR5" can work.
        """
        return self.spliced_length

    def __hasitem__(self, pos):
        """check if the coordinate falls into an exon"""
        try:
            x = self.map_to_spliced(pos)
        except ValueError:
            return False
        else:
            return True

    def EmptyChain(self):
        return ExonChain(self.chrom, self.strand, [0], [0], genome=self.genome)


class Transcript(ExonChain):
    def __init__(
        self,
        name,
        chrom,
        sense,
        exon_starts,
        exon_ends,
        cds,
        score=0,
        genome=None,
        description={},
        **kwargs,
    ):
        # Initialize underlying ExonChain
        super(Transcript, self).__init__(
            chrom, sense, exon_starts, exon_ends, genome=genome
        )
        self.score = score
        self.category = "transcript/body"
        self.transcript_id = kwargs.get("transcript_id", name)
        self.gene_id = kwargs.get("gene_id", name)
        self.gene_name = kwargs.get("gene_name", name)
        self.gene_type = kwargs.get("gene_type", "unknown")

        self.name = name
        self.description = description

        cds_start, cds_end = cds
        if cds_start == cds_end:
            self.UTR5 = None
            self.UTR3 = None
            self.CDS = None
            self.coding = False
            self.tx_class = description.get("tx_class", "ncRNA")
        else:
            self.UTR5, self.CDS, self.UTR3 = self.intersect(
                ["UTR5", "CDS", "UTR3"], cds_start, cds_end
            )
            if not (self.CDS.start == cds_start and self.CDS.end == cds_end):
                print(self.gene_name, self.name)
                print(self.CDS)
                print(cds_start, cds_end)
                1 / 0

            self.coding = True
            self.tx_class = description.get("tx_class", "coding")
            if self.UTR5:
                self.UTR5.name = "%s/UTR5" % self.name
                self.UTR5.gene_id = self.gene_id
                self.UTR5.category = "transcript/UTR5"

            if self.CDS:
                self.CDS.name = "%s/CDS" % self.name
                self.CDS.gene_id = self.gene_id
                self.UTR5.category = "transcript/CDS"

            if self.UTR3:
                self.UTR3.name = "%s/UTR3" % self.name
                self.UTR3.gene_id = self.gene_id
                self.UTR5.category = "transcript/UTR3"

            if self.strand == "+":
                self.CDS.start_codon = cds_start
                self.CDS.stop_codon = cds_end
            else:
                self.CDS.start_codon = cds_end
                self.CDS.stop_codon = cds_start

    @classmethod
    def from_GTF(cls, fname="", ORF_thresh=20, genome=None):
        shared_attributes = {}
        exon_starts = []
        exon_ends = []
        chrom = "NN"
        sense = "NN"
        current_tx = "NN"

        n = 0
        cds_min = np.inf
        cds_max = -np.inf
        ignore_records = set(["gene", "start_codon", "stop_codon", "UTR", "transcript"])
        for i, gff in enumerate(GFF_importer(fname)):
            # sys.stderr.write(".")
            if gff.type in ignore_records:
                continue

            attrs = attr_to_dict(gff.attr_str)
            tx_id = attrs.get("transcript_id", "unknown_tx_{}".format(n))
            if tx_id != current_tx:
                if exon_starts:
                    if cds_min is np.inf:
                        cds = (exon_starts[0], exon_starts[0])
                    else:
                        cds = (cds_min, cds_max)

                    yield cls(
                        current_tx,
                        chrom,
                        sense,
                        exon_starts,
                        exon_ends,
                        cds,
                        genome=genome,
                        **shared_attributes,
                    )

                exon_starts = []
                exon_ends = []
                shared_attributes = {}
                cds_min = np.inf
                cds_max = -np.inf
                n += 1

                current_tx = tx_id
                chrom = gff.chrom
                sense = gff.strand

            if gff.type == "CDS":
                cds_min = min(gff.start - 1, cds_min)
                cds_max = max(gff.end, cds_max)
                # print("encountered CDS record. current CDS", cds_min, cds_max)
                continue

            if gff.type != "exon":
                continue

            shared_attributes.update(attrs)

            exon_starts.append(gff.start - 1)
            exon_ends.append(gff.end)

        if exon_starts:
            if cds_min is np.inf:
                cds = (exon_starts[0], exon_starts[0])
            else:
                cds = (cds_min, cds_max)

            yield cls(
                tx_id,
                chrom,
                sense,
                exon_starts,
                exon_ends,
                cds,
                description=shared_attributes,
                genome=genome,
                **shared_attributes,
            )

    @classmethod
    def from_bed12(cls, line, genome=None):
        (
            chrom,
            start,
            end,
            name,
            score,
            strand,
            cds_start,
            cds_end,
            color,
            exon_count,
            block_sizes,
            block_starts,
        ) = line.rstrip().split("\t")

        start = int(start)
        end = int(end)

        exon_starts = np.array(block_starts.split(","), dtype=int) + start
        exon_ends = np.array(block_sizes.split(","), dtype=int) + exon_starts

        chain = cls(
            name,
            chrom,
            strand,
            exon_starts,
            exon_ends,
            (int(cds_start), int(cds_end)),
            score=np.nan if score == "." else float(score),
            genome=genome,
            description=dict(color=color, dtype=np.uint8),
        )

        return chain

    # UCSC table format output
    def __str__(self):
        return self.ucsc_format()


def _maybe_file(src):
    if hasattr(src, "read"):
        return src
    else:
        return open(src)


def from_bed6(src, genome=None):
    import numpy as np

    for line in _maybe_file(src):
        chrom, start, end, name, score, strand = line.rstrip().split("\t")
        chain = ExonChain(
            chrom,
            strand,
            [
                int(start),
            ],
            [
                int(end),
            ],
            genome=genome,
        )
        chain.name = name
        if score == ".":
            chain.score = np.nan
        else:
            chain.score = float(score)
        yield chain


def from_bed12(src, genome=None, filter=None):
    import numpy as np

    for line in _maybe_file(src):
        if not (filter is None) and not (filter in line):
            continue

        chain = Transcript.from_bed12(line, genome=genome)
        yield chain


class IndexedFasta(object):
    def __init__(self, fname, split_chrom="", **kwargs):
        import logging
        import mmap
        import os

        self.logger = logging.getLogger("scbamtools.gene_model.IndexedFasta")

        self.fname = fname
        self.chrom_stats = {}
        self.chrom_sizes = {}
        self.split_chrom = split_chrom

        f = open(fname)
        idx_file = f
        self._f = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

        # try to load index
        ipath = fname + ".byo_index"
        if os.access(ipath, os.R_OK):
            self.load_index(ipath)
        else:
            self.index(idx_file)
            self.store_index(ipath)

    def index(self, idx_file):
        self.logger.debug(
            "index('{self.fname}') split_chrom={self.split_chrom}".format(**locals())
        )

        ofs = 0
        chrom = "undef"
        chrom_ofs = 0
        size = 0
        nl_char = 0

        for line in idx_file:
            ofs += len(line)
            if line.startswith(">"):
                # store size of previous sequence
                if size:
                    self.chrom_stats[chrom].append(size)

                chrom = line[1:].split()[0].strip()
                if self.split_chrom:
                    # use this to strip garbage from chrom/contig name like chr1:new
                    # ->split_chrom=':' -> chrom=chr1
                    chrom = chrom.split(self.split_chrom)[0]
                chrom_ofs = ofs
            else:
                if not chrom in self.chrom_stats:
                    # this is the first line of the new chrom
                    size = 0
                    lline = len(line)
                    ldata = len(line.strip())
                    nl_char = lline - ldata
                    self.chrom_stats[chrom] = [chrom_ofs, ldata, nl_char, line[ldata:]]
                size += len(line.strip())

        # store size of previous sequence
        if size:
            self.chrom_stats[chrom].append(size)
        idx_file.flush()
        idx_file.seek(0)

    def store_index(self, ipath):
        import os

        self.logger.info("store_index('%s')" % ipath)

        # write to tmp-file first and in the end rename in order to have this atomic
        # otherwise parallel building of the same index may screw it up.

        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=os.path.dirname(ipath), delete=False
        )
        for chrom in sorted(self.chrom_stats.keys()):
            ofs, ldata, skip, skipchar, size = self.chrom_stats[chrom]
            tmp.write(
                "%s\t%d\t%d\t%d\t%r\t%d\n" % (chrom, ofs, ldata, skip, skipchar, size)
            )

        # make sure everything is on disk
        os.fsync(tmp)
        tmp.close()

        # make it accessible to everyone
        import stat

        os.chmod(tmp.name, stat.S_IROTH | stat.S_IRGRP | stat.S_IRUSR)

        # this is atomic on POSIX as we have created tmp in the same directory,
        # therefore same filesystem
        os.rename(tmp.name, ipath)

    def load_index(self, ipath):
        self.logger.info("load_index('%s')" % ipath)
        self.chrom_stats = {}
        for line in open(ipath):
            chrom, ofs, ldata, skip, skipchar, size = line.rstrip().split("\t")
            self.chrom_stats[chrom] = (
                int(ofs),
                int(ldata),
                int(skip),
                skipchar[1:-1].encode("ascii").decode("unicode_escape"),
                int(size),
            )
            self.chrom_sizes[chrom] = int(size)

    def get_data(self, chrom, start, end, sense):
        from scbamtools.util import rev_comp

        if not self.chrom_stats:
            self.index()

        ofs, ldata, skip, skip_char, size = self.chrom_stats[chrom]
        # print("ldata",ldata)
        # print("chromstats",self.chrom_stats[chrom])
        pad_start = 0
        pad_end = 0
        if start < 0:
            pad_start = -start
            start = 0

        if end > size:
            pad_end = end - size
            end = size

        l_start = int(start / ldata)
        l_end = int(end / ldata)
        # print "lines",l_start,l_end
        ofs_start = l_start * skip + start + ofs
        ofs_end = l_end * skip + end + ofs
        # print("ofs",ofs_start,ofs_end,ofs_end - ofs_start)
        # print(type(skip_char))

        s = self._f[ofs_start:ofs_end].decode("ascii").replace(skip_char, "")
        if pad_start or pad_end:
            s = "N" * pad_start + s + "N" * pad_end

        if sense == "-":
            s = rev_comp(s)
        return s


if __name__ == "__main__":
    genome = IndexedFasta("/data/rajewsky/genomes/GRCh38/GRCh38.fa")
    for tx in Transcript.from_GTF("/dev/stdin", genome=genome):
        print(tx.upstream(100).bed12_format())
        print(tx.downstream(100).bed12_format())
