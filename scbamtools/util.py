import logging
import os

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
        default=f"{prog}.log",
        help=f"place log entries in this file (default={prog}.log)",
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
