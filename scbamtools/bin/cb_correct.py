#!/usr/bin/env python3
from scbamtools.contrib import __version__, __author__, __license__, __email__
import logging
import os
import sys
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from time import time
import scbamtools.util as util
import scbamtools.cython.fastquery as fq

# from scbamtools import bcindex as bcindex
import mrfifo as mf


import logging
from time import time

# logging.basicConfig(level=logging.DEBUG)


class BCIndex:
    def __init__(self, path="bcref.mmap"):
        self.logger = logging.getLogger("BCIndex")
        self.MMAP = np.memmap(f"{path}", mode="r", dtype=np.uint32)
        assert (
            self.MMAP[:4]
            == np.array([ord("B"), ord("C"), ord("I"), 1], dtype=np.uint32)
        ).all(), "invalid BCI file"

        self.l_prefix = self.MMAP[4]
        self.l_suffix = self.MMAP[5]
        self.l = self.l_prefix + self.l_suffix
        self.n_prefixes = 2 ** (self.l_prefix * 2)
        self.n_seqs = self.MMAP[6]
        self.n_lists = self.MMAP[7]
        self.n_suffixes = self.MMAP[8]

        n_header = (
            9  # 'BCI' + version + l_prefix + l_suffix + n_seqs + n_lists + n_suffixes
        )
        n_prefix = 2 ** (self.l_prefix * 2)
        n_slists = (
            self.n_suffixes + self.n_lists
        )  # one uint32 for length of each suffix list
        n_total = n_header + n_prefix + n_slists

        self.PI = self.MMAP[n_header : n_header + self.n_prefixes]
        self.SL = self.MMAP[n_header + self.n_prefixes :]

        self.logger.debug(f"loaded {n_total *4} bytes packed references via MMAP:")
        self.logger.debug(
            f"  - prefix length: {self.l_prefix} suffix length: {self.l_suffix}"
        )
        self.logger.debug(f"  - {self.n_seqs} barcodes")
        self.logger.debug(f"  - {4 * n_total/self.n_seqs:.1f} bytes/barcode")
        self.logger.debug(
            f"  - {self.n_lists}/{self.n_prefixes} suffix lists (density: {self.n_lists/self.n_prefixes:.2f})"
        )
        self.logger.debug(
            f"  - {self.n_seqs/self.n_lists:.1f} average suffixes per list (expected recursion depth {np.log2(self.n_seqs/self.n_lists):.1f})"
        )

    @classmethod
    def build_from_barcodes(cls, src, mmap_path="bcref.mmap", l_prefix=10, l_suffix=0):
        # prefix index -> list index
        logger = logging.getLogger("BCIndex.build_from_barcodes")
        logger.debug("building temp set list")

        sets = defaultdict(set)

        # [set() for i in range(2 ** (l_prefix * 2))]

        logger.debug("* ingesting sequences")
        n_seqs = 0
        rshift = 2 * l_suffix
        mask = np.uint64((1 << (2 * l_suffix)) - 1)
        T0 = time()
        for seq in src:
            if l_suffix == 0:
                # automatic suffix length detection
                l_suffix = len(seq) - l_prefix
                logger.debug(
                    f"automatically setting l_prefix={l_prefix} to accomodate barcodes of length {len(seq)}"
                )
                assert l_suffix > 0
                rshift = 2 * l_suffix
                mask = np.uint64((1 << (2 * l_suffix)) - 1)

            idx = fq.seq_to_uint64(seq)
            n_seqs += 1
            idx_p = idx >> rshift
            idx_s = idx & mask

            sets[idx_p].add(idx_s)

            if n_seqs % 1000000 == 0:
                dT = time() - T0
                rate = n_seqs / dT / 1000
                logger.info(
                    f"... processed {n_seqs} sequences in {dT:.1f} seconds ({rate:.1f}k/sec)"
                )

        assert len(seq) == l_prefix + l_suffix

        dT = time() - T0
        rate = n_seqs / dT / 1000
        logger.info(
            f"* Finished ingest. Processed {n_seqs} sequences in {dT:.1f} seconds ({rate:.1f}k/sec)"
        )

        logger.debug("... computing sizes for packed suffix lists")
        n_lists = 0
        n_suffixes = 0
        for idx_p, S in sets.items():
            n_lists += 1
            n_suffixes += len(S)

        logger.debug(
            f"... allocating buffers for {n_lists} suffix lists with total of {n_suffixes} suffixes"
        )

        n_header = (
            9  # 'BCI' + version + l_prefix + l_suffix + n_seqs + n_lists + n_suffixes
        )
        n_prefix = 2 ** (l_prefix * 2)
        n_slists = n_suffixes + n_lists  # one uint32 for length of each suffix list
        n_total = n_header + n_prefix + n_slists
        logger.debug(f"* total MMAP size: {n_total *4} bytes")

        MMAP = np.memmap(mmap_path, mode="w+", shape=n_total + 1, dtype=np.uint32)
        MMAP[0:4] = np.array([ord("B"), ord("C"), ord("I"), 1], dtype=np.uint32)
        MMAP[4] = np.uint32(l_prefix)
        MMAP[5] = np.uint32(l_suffix)
        MMAP[6] = n_seqs
        MMAP[7] = n_lists
        MMAP[8] = n_suffixes

        PI = MMAP[n_header : n_header + n_prefix]
        SL = MMAP[n_header + n_prefix :]

        logger.debug("* packing suffix lists")
        ofs = 1  # one buffer byte, and keep 0 as "no list"
        for idx_p, S in sets.items():

            # compile the suffix set into a packed suffix list
            sl = sorted(sets[idx_p])
            L = len(sl)
            PI[idx_p] = ofs
            SL[ofs] = L
            ofs += 1
            SL[ofs : ofs + L] = sl  # assign values
            ofs += L

        logger.debug(f"needed {ofs *4} bytes for packed suffix lists")

        # load and open MMAPed index
        bci = cls(mmap_path)
        return bci

    def dump(self):
        for idx_p in np.arange(self.L_idx_p):
            l_ofs = self.PI[idx_p]
            if l_ofs > 0:
                prefix = _to_seq(idx_p, self.l_prefix)
                n = self.SL[l_ofs]  # number of suffixes to expect
                l_ofs += 1
                for i in range(n):
                    idx_s = self.SL[l_ofs + i]
                    suffix = _to_seq(idx_s, self.l_suffix)
                    yield f"{prefix}{suffix}"

    def query_idx64(self, bc_list):
        logger = logging.getLogger("BCIndex.query_idx64()")
        T0 = time()
        hits = np.zeros(len(bc_list), dtype=np.bool)
        seqidx.query_idx64(
            bc_list, hits, self.PI, self.SL, self.l_prefix, self.l_suffix
        )
        dT = time() - T0
        rate = len(bc_list) / dT / 1000
        logger.debug(f"queried {len(bc_list)} in {dT:.1f} seconds ({rate:.2f} k/sec)")

        return hits


def reader(fname, n_max=None):
    from time import time

    n = 0
    if fname.endswith(".gz"):
        logging.debug("opening gzip file")
        import isal.igzip_threaded as igzip

        _open = igzip.open
    else:
        _open = open

    T0 = time()
    for line in _open(fname, "rt"):
        if line.startswith("cell_bc"):
            continue

        n += 1
        if n_max and n > n_max:
            break

        seq = line.rstrip().split("\t", maxsplit=1)[0]  # .replace("N", "A")
        # yield fq.seq_to_uint64(bytes(seq, "ascii"))
        yield (bytes(seq, "ascii"))

    dT = time() - T0
    rate = n / dT / 1000
    logging.debug(f"read {n} barcodes in {dT:.1f} seconds ({rate:.2f} k/sec)")


def make_edit_dict(l=25):
    d = {
        0: "X",  # no match
        1: "=",  # exact match
    }

    i = 2

    # insertions first
    for pos in range(l):
        for b in "ACGT":
            if pos < l - 1:
                d[i] = f"I{l-pos}{b}1"
                i += 1
            if pos > 0:
                d[i] = f"I{l-pos}{b}0"
                i += 1

    # substitutions
    for pos in range(l):
        for k in range(1, 4):
            d[i] = f"S{l-pos}+{k}"
            i += 1

    # deletions
    for pos in range(l):
        for b in "ACGT":
            d[i] = f"_{l-pos}..{b}"
            i += 1
            d[i] = f"_{b}..{l-pos}"
            i += 1

    return d


def build_index(args):
    bci = BCIndex.build_from_barcodes(
        reader(args.input),
        mmap_path=args.index,
        l_prefix=args.l_prefix,
        l_suffix=args.l_suffix,
    )
    return bci


def write_edit_stats(n_edits, args):
    logger = logging.getLogger("scbamtools.cb_correct.write_edit_stats")
    logger.debug(f"writing statistics to {args.stats_out}")

    with open(util.ensure_path(args.stats_out), "wt") as fout:
        fout.write("edit\tn\n")

        for e, n in sorted(n_edits.items(), key=lambda x: -x[1]):
            fout.write(f"{e}\t{n}\n")


def query_barcodes(args):
    logger = logging.getLogger("scbamtools.cb_correct.query_barcodes")
    logger.debug("loading barcode index")
    bci = BCIndex(path=args.index)

    logger.debug("loading queries and converting to uint64")
    T0 = time()
    idx_data = fq.load_and_unique_sorted_barcodes(
        args.input,
        k=bci.l,
        n_max=args.n_max,
        unique=args.unique,
        # buf_size=1024 * 16,
    )
    dT = time() - T0
    rate = len(idx_data) / dT / 1000
    logger.debug(
        f"loaded {len(idx_data)} barcodes in {dT:.1f} seconds ({rate:.2f} k/sec)"
    )

    if args.dist == "0":
        logger.info("querying barcodes for exact matches (d=0)")
        hit_flag = np.zeros(len(idx_data), dtype=np.uint8)
        T0 = time()
        fq.query_idx64(
            idx_data,
            hit_flag,
            bci.PI,
            bci.SL,
            bci.l_prefix,
            bci.l_suffix,
        )

        dT = time() - T0
        rate = len(idx_data) / dT / 1000
        logger.info(
            f"queried {len(idx_data)} barcodes in {dT:.1f} seconds ({rate:.2f} k/sec)"
        )
        n_total_hits = (hit_flag > 0).sum()
        logger.info(
            f"found {n_total_hits}/{len(idx_data)} hits (match-rate = {n_total_hits/len(idx_data):.4f})"
        )
        hits = idx_data.copy()
        hit_variants = np.zeros(hits.shape, dtype=np.int16)
        hit_variants[:] = hit_flag[:]  # 1 -> '=' exact match
    else:
        logger.debug("querying barcodes")
        hit_variants = np.zeros(len(idx_data), dtype=np.int16)
        hits = np.zeros(len(idx_data), dtype=np.uint64)
        T0 = time()
        n_total_queries = fq.query_idx64_variants_omp(
            idx_data,
            hits,
            hit_variants,
            bci.PI,
            bci.SL,
            bci.l_prefix,
            bci.l_suffix,
            n_threads=args.threads,
        )

        dT = time() - T0
        rate = len(idx_data) / dT / 1000
        logger.info(
            f"queried {len(idx_data)} barcodes in {dT:.1f} seconds ({rate:.2f} k/sec)"
        )
        n_total_hits = (hit_variants > 0).sum()
        logger.info(
            f"found {n_total_hits}/{len(idx_data)} hits (match-rate = {n_total_hits/len(idx_data):.4f})"
        )

        logger.info(
            f"total queries processed: {n_total_queries} ({n_total_queries/len(idx_data):.2f} per barcode) rate: {n_total_queries / dT / 1000:.2f} k/sec)"
        )

    d = make_edit_dict(l=bci.l)
    if args.stats_out:
        n_edits = defaultdict(int)
        for i, n in enumerate(np.bincount(hit_variants)):
            n_edits[d[i]] = n

        write_edit_stats(n_edits, args)

    logger.info(f"writing results to {args.output}")
    with open(util.ensure_path(args.output), "wt") as fout:
        if args.out_mode == "table":
            fout.write("barcode\tmatch\tedit\n")
            for i in range(len(idx_data)):
                bc = fq.uint64_to_seq(idx_data[i], bci.l)
                match_bc = fq.uint64_to_seq(hits[i], bci.l)
                edit = d[hit_variants[i]]
                fout.write(f"{bc}\t{match_bc}\t{edit}\n")

        elif args.out_mode == "match":
            for bc in idx_data[hit_variants == 1]:
                match_bc = fq.uint64_to_seq(bc, bci.l)
                fout.write(f"{match_bc}\n")


def process_sam_records_serial(input, output, args):
    logger = util.setup_logging(args, name="scbamtools.cb_correct.process_sam_records")
    logger.debug("loading barcode index")
    bci = BCIndex(path=args.index)

    d = make_edit_dict(l=bci.l)

    edit_counts = defaultdict(int)

    # main iteration over SAM records
    for line in input:
        # find CB tag
        fields = line.rstrip().split("\t")
        cb_tag = None
        for field in fields[11:]:
            if field.startswith("CB:Z:"):
                cb_tag = field[5:]
                break

        if cb_tag:
            # we have a CB tag, try to find or correct it
            hit, edit_code = fq.query_idx64_variants_single(
                fq.seq_to_uint64(bytes(cb_tag, "ascii")),
                bci.PI,
                bci.SL,
                bci.l_prefix,
                bci.l_suffix,
            )
            edit_str = d[edit_code]
            edit_counts[edit_str] += 1

            if edit_code > 0:
                # found a match, update CB tag
                corrected_cb = fq.uint64_to_seq(hit, bci.l)
                line = line.replace(
                    f"CB:Z:{cb_tag}", f"CB:Z:{corrected_cb}\tcb:Z:{edit_str}"
                )

        # one line goes out for every line that comes in
        output.write(line)

    return edit_counts


# def process_sam_records(input, output, args):
#     logger = util.setup_logging(args, name="scbamtools.cb_correct.process_sam_records")
#     logger.debug("loading barcode index")
#     bci = BCIndex(path=args.index)

#     d = make_edit_dict(l=bci.l)

#     batch_size = 2000000
#     batch = []
#     cb_batch = []

#     edit_counts = defaultdict(int)

#     def process_batch(batch, cb_batch):
#         # convert CBs to uint64
#         cb_idxs = np.array(
#             [fq.seq_to_uint64(bytes(cb, "ascii")) for cb in cb_batch],
#             dtype=np.uint64,
#         )

#         # query index
#         hit_variants = np.zeros(len(cb_idxs), dtype=np.int16)
#         hits = np.zeros(len(cb_idxs), dtype=np.uint64)
#         fq.query_idx64_variants_omp(
#             cb_idxs,
#             hits,
#             hit_variants,
#             bci.PI,
#             bci.SL,
#             bci.l_prefix,
#             bci.l_suffix,
#             n_threads=args.threads,
#         )

#         for i, (line, cb_tag) in enumerate(zip(batch, cb_batch)):
#             hit_variant = hit_variants[i]
#             hit = hits[i]

#             edit = d[hit_variant]
#             edit_counts[edit] += 1
#             if hit_variant > 0:
#                 # found a match, update CB tag
#                 corrected_cb = fq.uint64_to_seq(hit, bci.l)
#                 line = line.replace(
#                     f"CB:Z:{cb_tag}", f"CB:Z:{corrected_cb}\tcb:Z:{edit}"
#                 )

#                 output.write(line)
#             else:
#                 pass
#                 # no match, pass through
#                 # output_fifo.write(line)

#     # main iteration over SAM records
#     for line in input:
#         if line.startswith("@"):
#             # header line, pass through
#             output.write(line)
#             continue

#         fields = line.rstrip().split("\t")
#         # find CB tag
#         cb_tag = None
#         for field in fields[11:]:
#             if field.startswith("CB:Z:"):
#                 cb_tag = field[5:]
#                 break

#         if cb_tag is None:
#             # no CB tag, pass through
#             # output_fifo.write(line)
#             continue

#         batch.append(line)
#         cb_batch.append(cb_tag)

#         if len(batch) >= batch_size:
#             process_batch(batch, cb_batch)
#             batch = []
#             cb_batch = []

#     # process remaining records
#     if len(batch) > 0:
#         process_batch(batch, cb_batch)

#     return edit_counts


def output_tee(
    input, output_match, output_nomatch=None, policy="same", rate_every=1000000
):
    logger = logging.getLogger("scbamtools.cb_correct.output_tee")
    N = defaultdict(int)

    T0 = time()
    for line in input:
        if line.startswith("@"):
            # header line, pass through
            output_match.write(line)
            if output_nomatch:
                output_nomatch.write(line)

            continue

        N["total"] += 1
        if "\tcb:Z:" in line:
            N["match"] += 1
            output_match.write(line)
        else:
            N["nomatch"] += 1
            if output_nomatch:
                output_nomatch.write(line)
            elif policy == "same":
                output_match.write(line)
            else:
                N["discarded"] += 1

        if N["total"] % rate_every == 0:
            dT = time() - T0
            rate = N["total"] / dT / 1000
            logger.info(
                f"... processed {N['total']} records (match_rate={100.0 * N['match']/N['total']:.1f} %) in {dT:.1f} seconds ({rate:.2f} k/sec)"
            )

    return N


def correct_cram(args):
    logger = util.setup_logging(args, name="scbamtools.cb_correct.correct_cram")
    logger.debug("loading barcode index")

    w = (
        mf.Workflow("cb")
        .BAM_reader(input=args.input, threads=args.threads_read)
        .distribute(
            input=mf.FIFO("input_sam", "rt"),
            outputs=mf.FIFO("dist{n}", "wt", n=args.threads),
            chunk_size=1,
            header_detect_func=util.is_header,
            header_broadcast=False,
            header_fifo=mf.FIFO("orig_header", "wt"),
        )
        .workers(
            input=mf.FIFO("dist{n}", "rt"),
            output=mf.FIFO("out{n}", "wt"),
            func=process_sam_records_serial,
            args=args,
            n=args.threads,
        )
        .add_job(
            func=util.update_header,
            input=mf.FIFO("orig_header", "rt"),
            output=mf.FIFO("new_header", "wt"),
            progname="cb_correct.py",
        )
        .collect(
            inputs=mf.FIFO("out{n}", "rt", n=args.threads),
            header_fifo=mf.FIFO("new_header", "rt"),
            output=mf.FIFO("out_sam", "wt"),
            chunk_size=1,
        )
    )
    if args.nomatch_out not in ["discard", "same"]:
        w.add_job(
            func=output_tee,
            input=mf.FIFO("out_sam", "rt"),
            output_match=mf.FIFO("out_match_sam", "wt"),
            output_nomatch=mf.FIFO("out_nomatch_sam", "wt"),
        )
        w.BAM_writer(
            input=mf.FIFO("out_nomatch_sam", "rt"),
            output=args.nomatch_out,
        )
    else:
        w.add_job(
            func=output_tee,
            job_name="cb.output_tee",
            input=mf.FIFO("out_sam", "rt"),
            output_match=mf.FIFO("out_match_sam", "wt"),
            policy=args.nomatch_out,
        )

    w.BAM_writer(
        input=mf.FIFO("out_match_sam", "rt"),
        output=args.bam_out,
        threads=args.threads_write,
        fmt=f"Sh{args.bam_out_mode}",
    )

    w.run()

    if args.stats_out:
        # collect and write stats
        # print(w.result_dict)
        edit_counts = mf.util.aggregate_dicts(w)
        # print(edit_counts)

        write_edit_stats(edit_counts, args)


def parse_args():
    import scbamtools.util as util

    parser = util.make_minimal_parser(
        "cb_correct.py",
        description="correct cell barcodes in SAM/BAM/CRAM streams based on a reference list",
    )
    # parser.set_defaults(func=parser.print_help)
    subparsers = parser.add_subparsers()

    # build index
    index_parser = subparsers.add_parser("index")
    index_parser.set_defaults(func=build_index)
    index_parser.add_argument(
        "--input",
        default="/dev/stdin",
        help="path to a simple flat file (possibly gzipped) with barcodes to be queried (default=stdin)",
    )
    index_parser.add_argument(
        "--index",
        default="bcref.mmap",
        help="filename for the compiled barcode index (default=bcref.mmap)",
    )
    index_parser.add_argument(
        "--l-prefix",
        default=10,
        type=int,
        help="number of barcode bases used as prefix for indexing (default=10)",
    )
    index_parser.add_argument(
        "--l-suffix",
        default=0,
        type=int,
        help="number of barcode bases used as suffix for indexing (default=0 -> L(barcode) - l_prefix)",
    )
    index_parser.add_argument(
        "--force-overwrite",
        default=False,
        action="store_true",
        help="if barcode index file exists, overwrite it (default=False)",
    )

    # correct
    correct_parser = subparsers.add_parser("sam")
    correct_parser.set_defaults(func=correct_cram)
    correct_parser.add_argument(
        "--input",
        default="/dev/stdin",
        help="path to a simple flat file (possibly gzipped) with barcodes to be queried (default=stdin)",
    )
    correct_parser.add_argument(
        "--index",
        default="bcref.mmap",
        help="filename for the compiled barcode index (default=bcref.mmap)",
    )
    correct_parser.add_argument(
        "--bam-out",
        default="/dev/stdout",
        help="path for the CB-corrected BAM output (default=stdout)",
    )
    correct_parser.add_argument(
        "--nomatch-out",
        default="discard",
        help=(
            "path for the BAM records without match to the index (default='discard', meaning no output.'"
            "Other options are 'same' to write them to the same output as matched records,"
            " or a path to a separate BAM/CRAM file to write them to."
        ),
    )
    correct_parser.add_argument(
        "--bam-out-mode",
        default="C",
        help="mode of the output BAM file (default=C for CRAM)",
    )
    correct_parser.add_argument(
        "--stats-out", default="", help="path for statistics output"
    )
    correct_parser.add_argument(
        "--threads-read",
        help="number of threads for reading bam_in (default=2)",
        type=int,
        default=2,
    )
    correct_parser.add_argument(
        "--threads-write",
        help="number of threads for writing bam_out (default=4)",
        type=int,
        default=4,
    )
    correct_parser.add_argument(
        "--threads",
        help="number of worker threads for actual CB queries (default=8)",
        type=int,
        default=8,
    )

    # query
    query_parser = subparsers.add_parser("query")
    query_parser.set_defaults(func=query_barcodes)
    query_parser.add_argument(
        "--index",
        default="bcref.mmap",
        help="filename for the compiled barcode index (default=bcref.mmap)",
    )
    query_parser.add_argument(
        "--input",
        default="/dev/stdin",
        help="path to a simple flat file (possibly gzipped) with barcodes to be queried (default=stdin)",
    )
    query_parser.add_argument(
        "--n-max",
        default=0,
        type=int,
        help="DEBUG: stop after reading the first N queries (default=0, meaning no limit)",
    )
    query_parser.add_argument(
        "--unique",
        default=False,
        action="store_true",
        help="only consider unique barcodes in the input (default=False)",
    )
    query_parser.add_argument(
        "--exact-only",
        default=False,
        action="store_true",
        help="only consider exact matches (default=False)",
    )
    query_parser.add_argument(
        "--output",
        default="/dev/stdout",
        help="path to write the corrected barcodes to (default=stdout)",
    )
    query_parser.add_argument(
        "--stats-out",
        default="bc_query_stats.tsv",
        help="path to the simple flat file with barcodes to be queried (default=stdin)",
    )
    query_parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="number of parallel threads to use for querying (default=8)",
    )
    query_parser.add_argument(
        "--dist",
        default="1",
        choices=["0", "1"],  # TODO: "1.5" for extended d=2 at the end only
        help="disable (0) or enable (1) Levenshtein distance=1 matching (default=1)",
    )
    query_parser.add_argument(
        "--out-mode",
        default="table",
        choices=["table", "match"],
        help="what kind of output to report. 'table' (default) has query, match, edit as columns. 'match' only the match",
    )

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    return args


def cmdline():
    args = parse_args()
    logger = util.setup_logging(args, name="scbamtools.bin.cb_correct")
    return args.func(args)

    # return main(args)


if __name__ == "__main__":
    res = cmdline()
