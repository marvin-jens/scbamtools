from scbamtools.contrib import __version__, __license__, __author__, __email__
import os
import sys
import logging
from collections import defaultdict
import mrfifo as mf
import scbamtools.util as util
import scbamtools.config as config
from scbamtools.annotation import GenomeAnnotation, CompiledClassifier


class AnnotationStats:
    logger = logging.getLogger("scbamtools.annotator.stats")

    def __init__(
        self,
        mapq=defaultdict(int),
        flag=defaultdict(int),
        gn=defaultdict(int),
        gf=defaultdict(int),
        gt=defaultdict(int),
        **kw,
    ):
        from collections import defaultdict

        self.mapq = mapq
        self.flag = flag
        self.gn = gn
        self.gf = gf
        self.gt = gt

        self.last_qname = None

    def count(self, read, gn_val, gf_val, gt_val):
        # count each read only once even if we have multiple alignments or mate pairs
        if read.qname != self.last_qname:
            self.flag[read.flag] += 1
            self.mapq[read.mapping_quality] += 1
            self.gf[gf_val] += 1
            self.gt[gt_val] += 1
            self.gn[gn_val] += 1

        self.last_qname = read.qname

    def as_dict(self):
        return dict(
            mapq=self.mapq,
            flag=self.flag,
            gn=self.gn,
            gf=self.gf,
            gt=self.gt,
        )

    def save_stats(self, fname, fields=["mapq", "flag", "gt", "gf", "gn"]):
        import pandas as pd

        fname = util.ensure_path(fname)

        data = []
        for name in fields:
            d = getattr(self, name)
            for obs, count in sorted(d.items(), key=lambda x: -x[1]):
                data.append((name, obs, count))

        self.logger.debug(f"writing annotation statistics to '{fname}'")
        df = pd.DataFrame(data, columns=["field", "value", "count"])
        df.to_csv(fname, sep="\t", index=None)
        return df


def blocks_from_cigar(cigar, pos=0):
    import re

    start = pos
    for x, op in re.findall(r"(\d+)(\w)", cigar):
        if op == "M":
            pos += int(x)
        elif op == "D":
            pos += int(x)
        elif op == "N":
            yield (start, pos)
            pos += int(x)
            start = pos

    if pos > start:
        yield (start, pos)


def annotate_SAM(input, output, compiled_annotation):
    """_summary_
    Worker sub-process. Annotate SAM lines using compiled GenomeAnnotation
    and place complete gn, gf, gt string values.

    Args:
        compiled_annotation (str): path to directory with compiled annotation information
    """
    logger = logging.getLogger("scbamtools.annotator.annotate_SAM")
    logger.debug(
        f"reading from '{input}' writing to '{output}' compiled_annotation='{compiled_annotation}'"
    )
    ga = GenomeAnnotation.from_compiled_index(compiled_annotation)

    for sam_line in input:
        sam = sam_line.split("\t")
        flags = int(sam[1])
        is_unmapped = flags & 4
        gn = None
        gf = "-"
        gt = None
        if not is_unmapped:
            chrom = sam[2]
            pos = int(sam[3]) - 1  # SAM uses one-based coordinates!!!
            cigar = sam[5]
            strand = "-" if (flags & 16) else "+"
            gn, gf, gt = ga.get_annotation_tags(
                chrom, strand, blocks_from_cigar(cigar, pos=pos)
            )

        tag_str = f"\tgn:Z:{gn}\tgf:Z:{gf}\tgt:Z:{gt}\n"
        output.write(sam_line.replace("\n", tag_str))


def is_header(line):
    return line.startswith("@")


##############################################################
### Here are the main functions as offered via commandline ###
##############################################################


def main(args):
    """_summary_
    Main function of the 'annotate' command. Create the plumbing for parallel worker processes and a single
    collector/writer process.

    Args:
        args (namespace): the command-line arguments reported from the parser
    """
    logger = util.setup_logging(args, name="scbamtools.cutadapt_bam.main")

    w = (
        mf.Workflow("ann")
        .BAM_reader(input=args.bam_in, threads=args.threads_read)
        .distribute(
            input=mf.FIFO("input_sam", "rt"),
            outputs=mf.FIFO("dist{n}", "wt", n=args.threads_work),
            chunk_size=1,
            header_detect_func=is_header,
            header_broadcast=False,
            header_fifo=mf.FIFO("orig_header", "wt"),
        )
        .workers(
            input=mf.FIFO("dist{n}", "rt"),
            output=mf.FIFO("out{n}", "wt"),
            func=annotate_SAM,
            compiled_annotation=args.compiled,
            n=args.threads_work,
        )
        .add_job(
            func=util.update_header,
            input=mf.FIFO("orig_header", "rt"),
            output=mf.FIFO("new_header", "wt"),
            progname="ann.py",
        )
        .collect(
            inputs=mf.FIFO("out{n}", "rt", n=args.threads_work),
            header_fifo=mf.FIFO("new_header", "rt"),
            output=mf.FIFO("out_sam", "wt"),
            log_rate_every_n=1000000,
            log_rate_template="annotated {M_out:.1f} M SAM records ({mps:.3f} M/s, overall {MPS:.3f} M/s)",
            chunk_size=1,
        )
        .funnel(
            input=mf.FIFO("out_sam", "rt"),
            output=args.bam_out,
            _manage_fifos=False,
            func=mf.parts.bam_writer,
            threads=args.threads_write,
            fmt=f"Sh{args.bam_out_mode}",
        )
        .run()
    )


def build_compiled_annotation(args):
    logger = logging.getLogger("scbamtools.annotator.build_compiled_annotation")
    if CompiledClassifier.files_exist(args.compiled):
        logger.warning(
            "already found a compiled annotation. use --force-overwrite to overwrite"
        )
        # ga = GenomeAnnotation.from_compiled_index(args.compiled)
        if not args.force_overwrite:
            return

    if args.tabular and os.access(args.tabular, os.R_OK):
        ga = GenomeAnnotation.from_uncompiled_df(args.tabular)
    else:
        ga = GenomeAnnotation.from_GTF(args.gtf, df_cache=args.tabular)

    ga = ga.compile(args.compiled)


def query_regions(args):
    logger = logging.getLogger("scbamtools.annotator.query_regions")
    if args.compiled:
        ga = GenomeAnnotation.from_compiled_index(args.compiled)
    else:
        ga = GenomeAnnotation.from_GTF(args.gtf)

    for region in args.region:
        logger.debug(f"querying region '{region}'")
        chrom, coords, strand = region.split(":")
        start, end = coords.split("-")

        gn, gf, gt = ga.get_annotation_tags(
            chrom,
            strand,
            [
                (int(start), int(end)),
            ],
        )
        print(f"gn={gn}\tgf={gf}\tgt={gt}")


def query_gff(args):
    logger = logging.getLogger("scbamtools.annotator.query_gff")
    if args.compiled:
        ga = GenomeAnnotation.from_compiled_index(args.compiled)
    else:
        ga = GenomeAnnotation.from_GTF(args.gtf)

    for line in open(args.gff):
        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) != 9:
            # not a GTF/GFF formatted line
            continue

        chrom, source, feature, start, end, score, strand, frame, attr_str = parts

        start = int(start) - 1
        end = int(end)

        logger.debug(f"querying region '{chrom}:{start}-{end}:{strand}'")

        gn, gf, gt = ga.get_annotation_tags(
            chrom,
            strand,
            [
                (start, end),
            ],
        )
        print(f'{line.rstrip()} gn "{gn}"; gf "{gf}"; gt "{gt}";')


def parse_args():
    parser = util.make_minimal_parser("ann.py")  # argparse.ArgumentParser()

    def usage(args):
        parser.print_help()

    parser.set_defaults(func=usage)
    parser.add_argument(
        "--config", default="", help="path to custom config-file (default='')"
    )

    subparsers = parser.add_subparsers()

    # build
    build_parser = subparsers.add_parser("build")
    build_parser.set_defaults(func=build_compiled_annotation)
    build_parser.add_argument(
        "--gtf",
        default=None,
        required=True,
        help="path to the original annotation (e.g. gencodev38.gtf.gz)",
    )
    build_parser.add_argument(
        "--compiled",
        default=None,
        help="path to a directoy in which a compiled version of the GTF is stored",
    )
    build_parser.add_argument(
        "--tabular",
        default="",
        help="path to a cache of the tabular version of the relevant GTF features (optional)",
    )
    build_parser.add_argument(
        "--force-overwrite",
        default=False,
        action="store_true",
        help="re-compile GTF and overwrite the pre-existing compiled annotation",
    )

    # tag
    tag_parser = subparsers.add_parser("tag")
    tag_parser.set_defaults(func=main)
    tag_parser.add_argument(
        "--compiled",
        default=None,
        help="path to a directoy in which a compiled version of the GTF is stored",
    )
    tag_parser.add_argument(
        "--bam-in", default="/dev/stdin", help="path to the input BAM (default=stdin)"
    )
    tag_parser.add_argument(
        "--bam-out",
        default="/dev/stdout",
        help="path for the tagged BAM output (default=stdout)",
    )
    tag_parser.add_argument(
        "--bam-out-mode", default="b", help="mode of the output BAM file (default=b)"
    )
    tag_parser.add_argument(
        "--stats-out", default="", help="path for statistics output"
    )
    tag_parser.add_argument(
        "--threads-read",
        help="number of threads for reading bam_in (default=2)",
        type=int,
        default=2,
    )
    tag_parser.add_argument(
        "--threads-write",
        help="number of threads for writing bam_out (default=4)",
        type=int,
        default=4,
    )
    tag_parser.add_argument(
        "--threads-work",
        help="number of worker threads for actual trimming (default=8)",
        type=int,
        default=8,
    )

    # query
    query_parser = subparsers.add_parser("query")
    query_parser.set_defaults(func=query_regions)
    query_parser.add_argument(
        "--compiled",
        default=None,
        help="path to a directoy in which a compiled version of the GTF is stored",
    )
    query_parser.add_argument(
        "--gtf",
        default=None,
        help="path to the original annotation (e.g. gencodev38.gtf.gz)",
    )
    query_parser.add_argument("region", default=[], help="region to query", nargs="+")

    # gff
    gff_parser = subparsers.add_parser("gff")
    gff_parser.set_defaults(func=query_gff)
    gff_parser.add_argument(
        "--compiled",
        default=None,
        help="path to a directoy in which a compiled version of the GTF is stored",
    )
    gff_parser.add_argument(
        "--gff",
        default=None,
        help="path to the gff file you want queried",
    )

    args = parser.parse_args()
    return config.load(args.config, args=vars(args))


def cmdline():
    args = parse_args()
    util.setup_logging(args, "scbamtools.annotator.cmdline")
    return args.func(args)


if __name__ == "__main__":
    ret_code = cmdline()
    sys.exit(ret_code)

    # import cProfile
    # cProfile.run("cmdline()", "prof_stats")
