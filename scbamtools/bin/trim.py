from scbamtools.contrib import __version__, __license__, __author__, __email__
import numpy as np
from collections import defaultdict
import mrfifo as mf
import scbamtools.util as util
import scbamtools.config as config


def parse_args():
    parser = util.make_minimal_parser(
        prog="trim.py",
        description="trim adapters from a BAM file using cutadapt",
    )
    parser.add_argument(
        "--config", default="", help="path to custom config-file (default='')"
    )
    parser.add_argument(
        "bam_in",
        help="bam input (default=stdin)",
        default="/dev/stdin",
        # nargs="+",
    )
    parser.add_argument(
        "--bam-out",
        help="bam output (default=stdout)",
        default="/dev/stdout",
    )
    parser.add_argument(
        "--bam-out-mode",
        help="bam output mode (default=b)",
        default="b",
    )
    parser.add_argument(
        "--flavor",
        help="name of the adapter flavor used to retrieve sequences and parameters from the config.yaml",
        default="default",
    )
    parser.add_argument(
        "--skim",
        help="skim through the BAM by investigating only every <skim>-th record (default=1 off)",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--phred-base",
        help="phred score base used for qual trimming (default=33)",
        type=int,
        default=33,
    )
    parser.add_argument(
        "--threads-read",
        help="number of threads for reading bam_in (default=2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--threads-write",
        help="number of threads for writing bam_out (default=4)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threads-work",
        help="number of worker threads for actual trimming (default=8)",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--stats-out",
        help="write tab-separated table with trimming results here",
        default="",
    )
    args = parser.parse_args()
    return config.load(args.config, args=vars(args))


class QualityTrim:
    def __init__(self, min_base_qual=20):
        self.min_base_qual = min_base_qual
        self.name = "Q"

    def match_to(self, seq, qual):
        qtrim = np.array(qual) >= self.min_base_qual
        n_trimmed = (qtrim[::-1]).argmax()
        return n_trimmed


class AdapterTrim:
    def __init__(self, name="na", where="right", seq=None, **kw):
        import cutadapt.adapters

        self.name = name
        self.where = where
        if where == "left":
            self.adapter = cutadapt.adapters.NonInternalFrontAdapter(
                seq, name=name, **kw
            )
        else:
            self.adapter = cutadapt.adapters.BackAdapter(seq, name=name, **kw)

    def match_to(self, seq, qual):
        match = self.adapter.match_to(seq)
        n_trimmed = 0
        if match:
            if self.where == "left":
                n_trimmed = match.rstop
            else:
                n_trimmed = len(seq) - match.rstart

        return n_trimmed


class AdapterFlavor:
    def __init__(
        self,
        args,
        default_max_errors=0.1,
        default_min_overlap=3,
        default_min_base_qual=20,
        default_min_read_length=18,
        default_paired_end="single_end",
    ):
        # import cutadapt.adapters

        self.stats = defaultdict(int)
        self.total = defaultdict(int)
        self.lhist = defaultdict(int)
        self.flavor = args.flavor
        if not args.flavor in args.adapter_flavors:
            raise KeyError(
                f"adapter_flavor '{args.flavor}' not found in config.yaml! Need valid --flavor=... "
            )

        flavor_d = args.adapter_flavors[self.flavor]
        # from pprint import pprint

        # pprint(flavor_d)
        self.adapter_sequences = args.adapters

        self.trimmers_right = []
        for adap_d in flavor_d.get("cut_right", []):
            for name, param_d in adap_d.items():
                if name == "Q":
                    adapter = QualityTrim(
                        min_base_qual=param_d.get(
                            "min_base_qual", default_min_base_qual
                        )
                    )
                else:
                    adapter = AdapterTrim(
                        name=name,
                        seq=self.adapter_sequences[name],
                        where="right",
                        max_errors=param_d.get("max_errors", default_max_errors),
                        min_overlap=param_d.get("min_overlap", default_min_overlap),
                    )

                self.trimmers_right.append(adapter)

        self.trimmers_left = []
        for adap_d in flavor_d.get("cut_left", []):
            for name, param_d in adap_d.items():
                adapter = AdapterTrim(
                    name=name,
                    seq=self.adapter_sequences[name],
                    where="left",
                    max_errors=param_d.get("max_errors", default_max_errors),
                    min_overlap=param_d.get("min_overlap", default_min_overlap),
                )
                self.trimmers_left.append(adapter)

        self.min_read_length = flavor_d.get("min_read_length", default_min_read_length)
        self.paired_end = flavor_d.get("paired_end", default_paired_end)

    def process_read(self, read_seq, read_qual):
        # print(read_qual)
        start = 0
        end = len(read_seq)

        self.stats["N_input"] += 1
        self.total["bp_input"] += end
        trimmed_names_right = []
        trimmed_bases_right = []

        trimmed_names_left = []
        trimmed_bases_left = []

        def check_discard(start, end, reason):
            if (end - start) < self.min_read_length:
                self.stats[reason] += 1
                self.stats["N_discarded"] += 1
                self.total["bp_discarded"] += end - start

                return True
            else:
                return False

        for trimmer in self.trimmers_right:
            n_trimmed = trimmer.match_to(read_seq[start:end], read_qual[start:end])
            if n_trimmed:
                end -= n_trimmed
                trimmed_bases_right.append(n_trimmed)
                trimmed_names_right.append(trimmer.name)

                self.stats[f"N_{trimmer.name}_trimmed"] += 1
                self.total[f"bp_{trimmer.name}_trimmed"] += n_trimmed
                self.total[f"bp_trimmed"] += n_trimmed

            if check_discard(start, end, f"N_too_short_after_{trimmer.name}"):
                return

        for trimmer in self.trimmers_left:
            n_trimmed = trimmer.match_to(read_seq[start:end], read_qual[start:end])
            if n_trimmed:
                start += n_trimmed
                trimmed_bases_left.append(n_trimmed)
                trimmed_names_left.append(trimmer.name)

                self.stats[f"N_{trimmer.name}_trimmed"] += 1
                self.total[f"bp_{trimmer.name}_trimmed"] += n_trimmed
                self.total[f"bp_trimmed"] += n_trimmed

            if check_discard(start, end, f"N_too_short_after_{trimmer.name}"):
                return

        # we've made it to the end!
        self.stats["N_kept"] += 1
        self.total["bp_kept"] += end

        tags = []
        if trimmed_names_right:
            tags.append(f"A3:Z:{','.join(trimmed_names_right)}")
            tags.append(f"T3:Z:{','.join([str(s) for s in trimmed_bases_right])}")

        if trimmed_names_left:
            tags.append(f"A5:Z:{','.join(trimmed_names_left)}")
            tags.append(f"T5:Z:{','.join([str(s) for s in trimmed_bases_left])}")

        self.lhist[end - start] += 1
        return start, end, "\t".join(tags)


def process_reads(input, output, args):
    flavor = AdapterFlavor(args)
    # TODO: paired-end processing
    for read in input:
        cols = read.split("\t")
        read_seq = cols[9]
        qual_str = cols[10]
        read_qual = np.array(bytearray(qual_str.encode("ASCII"))) - args.phred_base

        result = flavor.process_read(read_seq, read_qual)
        if result:
            start, end, tags = result
            cols[9] = read_seq[start:end]
            cols[10] = qual_str[start:end]

            if tags:
                cols[-1] = f"{cols[-1].rstrip()}\t{tags}\n"

            output.write("\t".join(cols))

    return {"stats": flavor.stats, "total": flavor.total, "lhist": flavor.lhist}


def main(args):
    logger = util.setup_logging(args, name="scbamtools.cutadapt_bam.main")

    w = (
        mf.Workflow("cutadapt")
        .BAM_reader(input=args.bam_in, threads=args.threads_read)
        .distribute(
            input=mf.FIFO("input_sam", "rt"),
            outputs=mf.FIFO("dist{n}", "wt", n=args.threads_work),
            chunk_size=1,
            header_detect_func=util.is_header,
            header_broadcast=False,
            header_fifo=mf.FIFO("orig_header", "wt"),
        )
        .workers(
            input=mf.FIFO("dist{n}", "rt"),
            output=mf.FIFO("out{n}", "wt"),
            func=process_reads,
            args=args,
            n=args.threads_work,
        )
        .add_job(
            func=util.update_header,
            input=mf.FIFO("orig_header", "rt"),
            output=mf.FIFO("new_header", "wt"),
            progname="trim.py",
        )
        .collect(
            inputs=mf.FIFO("out{n}", "rt", n=args.threads_work),
            header_fifo=mf.FIFO("new_header", "rt"),
            output=mf.FIFO("out_sam", "wt"),
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
    # stats = defaultdict(int)
    # total = defaultdict(int)
    # lhist = defaultdict(int)

    # dt = time() - t0
    # logger.info(
    #     f"processed {stats['N_input']} reads in {dt:.1f} seconds ({stats['N_input']/dt:.1f} reads/second)."
    # )

    # if args.stats_out:
    #     with open(util.ensure_path(args.stats_out), "wt") as f:
    #         f.write("key\tcount\tpercent\n")
    #         for k, v in sorted(stats.items(), key=lambda x: -x[1]):
    #             f.write(f"reads\t{k}\t{v}\t{100.0 * v/stats['N_input']:.2f}\n")

    #         for k, v in sorted(total.items(), key=lambda x: -x[1]):
    #             f.write(f"bases\t{k}\t{v}\t{100.0 * v/total['bp_input']:.2f}\n")

    #         for k, v in sorted(lhist.items()):
    #             f.write(f"L_final\t{k}\t{v}\t{100.0 * v/stats['N_kept']:.2f}\n")


def cmdline():
    args = parse_args()
    util.setup_logging(args)
    main(args)


if __name__ == "__main__":
    cmdline()
