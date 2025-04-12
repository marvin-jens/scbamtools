from scbamtools.contrib import __version__, __author__, __license__, __email__
import numpy as np
import sys
from collections import OrderedDict

import datetime
import os
import scbamtools.util as util
import scbamtools.config as config
from scbamtools.contrib import __version__, __author__, __license__
from scbamtools.quant import DGE, default_channels, sparse_summation
import mrfifo as mf


def cmdline():
    parser = util.make_minimal_parser(
        prog="count",
        description="quantify per-cell gene expression from BAM files by counting into a (sparse) DGE matrix",
        main=main,
    )
    parser.add_argument("--config", default="config.yaml", help="path to config-file")

    parser.add_argument(
        "bam_in",
        help="bam input (default=stdin)",
        default=["/dev/stdin"],
        nargs="+",
    )
    parser.add_argument(
        "--skim",
        help="skim through the BAM by investigating only every <skim>-th record (default=1 off)",
        default=1,
        type=int,
        # nargs="+",
    )
    parser.add_argument(
        "--worker-threads",
        help="number of parallel worker processes. (default=4)",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--input-threads",
        help="number of threads to use for BAM decompression. (default=2)",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--pipe-buffer-size",
        help="total pipe buffer size in MB to allocate via mrfifo for interprocess communication (default=16)",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--flavors",
        help="comma-separated list of quantification flavors to retrieve from the config.yaml and apply to each BAM/CRAM in the input. If not set, will use 'default' for every BAM/CRAM",
        default="",
    )
    parser.add_argument(
        "--cell-bc-allow-list",
        help="[OPTIONAL] a text file with cell barcodes. All barcodes not in the list are ignored.",
        default="",
    )
    parser.add_argument(
        "--output",
        help="directory to store the output h5ad and statistics/marginal counts",
        default="dge",
    )
    parser.add_argument(
        "--dge-out",
        help="filename for the output h5ad",
        default="{args.output}/{args.sample}.h5ad",
    )
    parser.add_argument(
        "--summary-out",
        help="filename for the output summary (sum over all vars/genes)",
        default="{args.output}/{args.sample}.summary.tsv",
    )
    parser.add_argument(
        "--bulk-out",
        help="filename for the output summary (sum over all vars/genes)",
        default="{args.output}/{args.sample}.pseudo_bulk.tsv",
    )
    parser.add_argument(
        "--stats-out",
        help="filename for the statistics/counts output",
        default="{args.output}/{args.sample}.stats.tsv",
    )
    args = parser.parse_args()
    return args


def get_counter_class(classpath):

    if "." in classpath:
        mod, cls = classpath.rsplit(".", 1)
        import importlib

        m = importlib.import_module(mod)
    else:
        m = globals().get(classpath)

    return getattr(m, cls)


def get_config_for_refs(args, default="default"):
    """
    Parse flavor definitions, filling a dictionary with reference name as key and configuration
    as values.
    The values are kwargs dictionaries, including the counter_class argument already replaced
    with the actual class object ready for instantiation.

    The key '*' is used for default settings.

    For convenience, we also collect the names of all channels known by all counter classes, so
    that the DGE object can be created before any data are parsed.
    """
    flavors = args.quant
    # replace str of counter class name/path with actual class object
    for name, config in flavors.items():
        config["name"] = name

    bam_d = {"*": flavors[default]}

    # collect all channel names that can be expected to be generated
    channels = list(default_channels)

    if args.flavors:
        for bam, flavor_name in zip(args.bam_in, args.flavors.split(",")):
            f = flavors[flavor_name]
            bam_d[bam] = f
            channels.extend(f.get("channels", default_channels))

    return bam_d, sorted(set(channels))


# def simple_check(input, output="/dev/stdout"):
#     n = 0
#     for line in input:
#         n += 1
#     return n


def sam_to_DGE(input, channels=[], cell_bc_allow_list="", config={}, **kw):
    import scbamtools.quant
    import logging

    logger = logging.getLogger("scbamtools.count.sam_to_DGE")
    logger.debug(
        f"using config {config['name']} -> counter_class = {config['counter_class']}"
    )
    counter_class = get_counter_class(
        config.get("counter_class", "scbamtools.quant.DefaultCounter")
    )
    # print(f"counter: {counter_class}")
    counter = counter_class(channels=channels, **config)

    dge = scbamtools.quant.DGE(channels=channels, cell_bc_allow_list=cell_bc_allow_list)
    for bundle in counter.bundles_from_SAM(input):
        cell, gene, channels = counter.process_bundle(bundle)
        if cell != "-":
            dge.add_read(cell, gene, channels)

    channel_d, obs_names, var_names = dge.make_sparse_arrays()
    # print("end of sam_to_DGE: channel_d", channel_d.keys(), obs_names, var_names)
    return {
        "channel_d": channel_d,
        "obs_names": obs_names,
        "var_names": var_names,
        "counter_stats": counter.stats,
    }


def main(args):
    from scbamtools.quant import DGE
    import anndata
    import pandas as pd
    from time import time

    qflavor_d, channels = get_config_for_refs(args)

    logger = util.setup_logging(args, f"scbamtools.count.main")
    logger.info(f"startup. Channels collected from config: '{channels}'")

    adata_refs = []
    util.ensure_path(args.output + "/")

    t0 = time()
    stats = []
    for input in args.bam_in:
        ref = os.path.basename(input).split(".")[0]
        # logger.info(f"processing alignments from {input} -> ref={ref}")
        if not input in qflavor_d:
            config = qflavor_d["*"]
            logger.warning(
                f"no config specified for input='{input}' (ref={ref}) ({sorted(qflavor_d.keys())}), falling back to default."
            )
        else:
            config = qflavor_d[input]

        logger.info(
            f"using quantification flavor '{config['name']}' for input file '{input}'"
        )
        w = (
            mf.Workflow("count", total_pipe_buffer_MB=args.pipe_buffer_size)
            .BAM_reader(
                input=input,
                threads=args.input_threads,
                output=mf.FIFO("input_sam", "w"),
            )
            .distribute(
                func=mf.parts.distribute_by_CB,
                input=mf.FIFO("input_sam", "r"),
                outputs=mf.FIFO("dist_{n}", "w", n=args.worker_threads),
                header_detect_func=mf.util.is_header,
                header_fifo="/dev/null",
                # sub_lead=b"\tCB:Z:" # TODO: modify to split by UMI, potentially having to add up counts?
            )
            .workers(
                func=sam_to_DGE,
                input=mf.FIFO("dist_{n}", "r"),
                channels=channels,
                cell_bc_allow_list=args.cell_bc_allow_list,
                n=args.worker_threads,
                config=config,
            )
            .run()
        )

        from collections import defaultdict

        counter_stats = defaultdict(int)
        counter_stats["ref"] = ref
        counter_stats["config"] = config["name"]

        # collect the counts and make one AnnData object
        adata_shards = []
        na_shards = []
        for name, res in w.result_dict.items():
            if name.startswith("count.worker"):
                if len(res["channel_d"]):
                    adata = DGE.sparse_arrays_to_adata(
                        res["channel_d"], res["obs_names"], res["var_names"]
                    )
                    # adata.obs["worker"] = name
                    if "NA" in adata.obs_names:
                        na_shards.append(adata["NA"])
                        na_mask = np.array([name == "NA" for name in adata.obs_names])
                        # drop the NA gene (for barcode outside allowlist) from
                        # adata and keep it separate
                        adata = adata[~na_mask].copy()

                    adata_shards.append(adata)

                for k, n in res["counter_stats"].items():
                    counter_stats[k] += n

        # compile one adata object for all barcode-prefix shards by concatenating on obs (cell barcodes)
        adata = anndata.concat(
            adata_shards, axis=0, join="outer", merge="unique", uns_merge="first"
        )
        adata.obs_names_make_unique()
        # # need to add up everything that is 'NA-x'
        # na_mask = np.array([o.startswith("NA-") for o in adata.obs_names])
        # na = adata[na_mask].X.sum(axis=0)
        # adata = adata[~na_mask].copy()
        # adata["NA"] = na
        adata.var["reference"] = ref
        adata.var["reference"] = pd.Categorical(adata.var["reference"])
        adata.obs[f"n_{ref}_counts"] = sparse_summation(adata.X, axis=1)
        adata_refs.append(adata)
        # adata.write_h5ad(f"{ref}.h5ad")

        stats.append(counter_stats)

    # compile one adata object for all BAM files by concatenating on vars (genes)
    adata = anndata.concat(
        adata_refs, axis=1, join="outer", merge="unique", uns_merge="first"
    )

    # number of molecules counted for this cell
    adata.obs[f"n_counts"] = sparse_summation(adata.X, axis=1)
    # number of genes detected in this cell
    adata.obs["n_genes"] = sparse_summation(adata.X > 0, axis=1)
    for channel in adata.layers.keys():
        adata.obs[f"n_{channel}"] = sparse_summation(adata.layers[channel], axis=1)
        adata.var[f"n_{channel}"] = sparse_summation(adata.layers[channel], axis=0)

    # total counts for a gene across all cells
    adata.var[f"n_counts"] = sparse_summation(adata.X, axis=0)
    # number of cells with at least one count for the gene
    adata.var[f"n_cells"] = sparse_summation(adata.X > 0, axis=0)

    adata.obs.index.name = "cell_bc"
    adata.var.index.name = "gene"
    adata.write_h5ad(args.dge_out)

    # store marginals
    adata.obs.sort_values("n_counts", ascending=False).to_csv(
        args.summary_out, sep="\t"
    )
    adata.var.sort_values("n_counts", ascending=False).to_csv(args.bulk_out, sep="\t")

    # store counting statistics
    df_stats = pd.DataFrame(stats)
    df_stats["sample"] = args.sample
    df_stats = df_stats[sorted(df_stats.columns)].fillna(0)
    # print(df_stats.T)
    df_stats.to_csv(args.stats_out, sep="\t", index=False)

    dt = time() - t0
    n_total = df_stats["SAM_records_total"].sum()
    logger.info(
        f"finished counting {n_total/1E6:.2f}M SAM records in {dt:.1f} seconds ({n_total/dt/1E6:.3f}M/sec)."
    )
    return adata


if __name__ == "__main__":
    args = cmdline()
    args.func(args)
