import logging
import os
import sys

default_log_level = "INFO"


def ensure_path(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    return path


def setup_logging(
    args,
    name="scbamtools.main",
    log_file="",
    FORMAT="%(asctime)-20s\t{sample:30s}\t%(name)-50s\t%(levelname)s\t%(message)s",
):
    sample = getattr(args, "sample", "na")
    import setproctitle

    if name != "scbamtools.main":
        setproctitle.setproctitle(f"{name} {sample}")

    FORMAT = FORMAT.format(sample=sample)

    log_level = getattr(args, "log_level", "INFO")
    lvl = getattr(logging, log_level)
    logging.basicConfig(level=lvl, format=FORMAT)
    root = logging.getLogger("spacemake")
    root.setLevel(lvl)

    log_file = getattr(args, "log_file", log_file)
    if log_file:
        fh = logging.FileHandler(filename=ensure_path(log_file), mode="a")
        fh.setFormatter(logging.Formatter(FORMAT))
        root.debug(f"adding log-file handler '{log_file}'")
        root.addHandler(fh)

    if hasattr(args, "debug"):
        # cmdline requested debug output for specific domains (comma-separated)
        for logger_name in args.debug.split(","):
            if logger_name:
                root.info(f"setting domain {logger_name} to DEBUG")
                logging.getLogger(logger_name.replace("root", "")).setLevel(
                    logging.DEBUG
                )

    logger = logging.getLogger(name)
    logger.debug("started logging")
    for k, v in sorted(vars(args).items()):
        logger.debug(f"cmdline arg\t{k}={v}")

    return logger


def make_minimal_parser(prog="", usage="", **kw):
    import argparse

    parser = argparse.ArgumentParser(prog=prog, usage=usage, **kw)
    parser.add_argument(
        "--log-file",
        default="",  # f"{prog}.log",
        help=f"place log entries in this file (default=off)",
    )
    parser.add_argument(
        "--log-level",
        default=default_log_level,
        help=f"change threshold of python logging facility (default={default_log_level})",
    )
    parser.add_argument(
        "--debug",
        default="",
        help=f"comma-separated list of logging-domains for which you want DEBUG output",
    )
    parser.add_argument(
        "--sample", default="sample_NA", help="sample_id (where applicable)"
    )
    return parser


def update_header(input, output, progname="scbamtools", cmdline=" ".join(sys.argv)):
    from collections import defaultdict
    from scbamtools.contrib import __version__

    id_counter = defaultdict(int)
    pp = ""
    for header_line in input:
        if header_line.startswith("@PG"):
            parts = header_line.split("\t")
            for part in parts[1:]:
                k, v = part.split(":", maxsplit=1)
                if k == "ID":
                    pp = v
                    id_counter[v] += 1

        output.write(header_line)

    pg_id = progname.split(".")[0]
    if id_counter[pg_id] > 0:
        pg_id += f".{id_counter[pg_id]}"

    output.write(
        f"@PG\tID:{pg_id}\tPN:{progname}\tPP:{pp}\tVN:{__version__}\tCL:{cmdline}\n"
    )


def generate_kmers(k, nts="ACGT"):
    if k == 0:
        yield ""
    elif k > 0:
        for x in nts:
            for mer in generate_kmers(k - 1, nts=nts):
                yield x + mer


COMPLEMENT = {
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
    "k": "m",
    "m": "k",
    "r": "y",
    "y": "r",
    "s": "s",
    "w": "w",
    "b": "v",
    "v": "b",
    "h": "d",
    "d": "h",
    "n": "n",
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "K": "M",
    "M": "K",
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "B": "V",
    "V": "B",
    "H": "D",
    "D": "H",
    "N": "N",
    "-": "-",
    "=": "=",
    "+": "+",
}


def complement(s):
    return "".join([COMPLEMENT[x] for x in s])


def rev_comp(seq):
    return complement(seq)[::-1]


def is_header(line):
    return line.startswith("@")
