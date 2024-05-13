from scbamtools.contrib import __version__, __author__, __license__, __email__
import numpy as np
import sys
from collections import OrderedDict

import datetime
import os
import scbamtools.util as util
import scbamtools.config as config

from scbamtools.quant import DGE, default_channels, sparse_summation
import mrfifo as mf


def parse_cmdline():
    parser = util.make_minimal_parser(
        prog="quant.py",
        description="quantify per-cell gene expression from BAM files by counting into a (sparse) DGE matrix",
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
        "--flavor",
        help="name of the adapter flavor used to retrieve sequences and parameters from the config.yaml. Can also be a mapping choosing specific flavor for each reference (example: 'first@miRNA,default@genome,chrom@rRNA')",
        default="default",
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
    return config.load(args.config, args=vars(args))


def get_counter_class(classpath):

    if "." in classpath:
        mod, cls = classpath.rsplit(".", 1)
        import importlib

        m = importlib.import_module(mod)
    else:
        m = globals().get(classpath)

    return getattr(m, cls)


def get_config_for_refs(args):
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
        config["counter_class"] = get_counter_class(
            config.get("counter_class", "scbamtools.quant.DefaultCounter")
        )
        config["name"] = name

    ref_d = {}
    default = "default"
    for f in args.flavor.split(","):
        if "@" in f:
            ref, name = f.split("@")
            ref_d[ref] = flavors[name]
        else:
            default = f

    ref_d["*"] = flavors[default]

    # collect all channel names that can be expected to be generated
    channels = []
    for ref, config in ref_d.items():
        channels.extend(config.get("channels", default_channels))

    return ref_d, sorted(set(channels))


# ## This will become the reader process
# def bam_iter_bundles(bam_src, logger, args, stats):
#     """
#     Generator: gathers successive BAM records into bundles as long as they belong to the same read/fragment/mate pair
#     (by qname). Recquired for chunking input data w/o breaking up this information.
#     """

#     def extract(rec):
#         return (
#             bam_src.header.get_reference_name(rec.tid),  # chrom
#             "-" if rec.is_reverse else "+",  # strand
#             rec.get_tag("gn").split(",") if rec.has_tag("gn") else ["-"],  # gene names
#             (
#                 rec.get_tag("gf").split(",") if rec.has_tag("gf") else ["-"]
#             ),  # gene feature overlap encodings
#             rec.get_tag("AS"),  # alignment score
#         )

#     last_qname = None
#     bundle = []
#     bundle_ref = None
#     CB = "NN"
#     MI = "NN"

#     # for ref, rec in util.timed_loop(bam_src, logger, skim=args.skim):
#     for bam_name, rec in bam_src:
#         ref = os.path.basename(bam_name).split(".")[0]

#         ref_stats = stats.stats_by_ref[ref]
#         ref_stats["N_records"] += 1
#         if rec.is_paired and rec.is_read2:
#             continue  # TODO: proper sanity checks and/or disambiguation

#         if rec.is_unmapped:
#             ref_stats["N_unmapped"] += 1
#             continue

#         if rec.query_name != last_qname:
#             ref_stats["N_frags"] += 1

#             if bundle:
#                 yield bundle_ref, CB, MI, bundle

#             CB = rec.get_tag("CB")
#             MI = rec.get_tag("MI")
#             bundle = [
#                 extract(rec),
#             ]
#             last_qname = rec.query_name
#             bundle_ref = ref
#         else:
#             bundle.append(extract(rec))

#     if bundle:
#         yield bundle_ref, CB, MI, bundle


# def bundle_processor(
#     Qin,
#     Qout,
#     Qerr,
#     abort_flag,
#     stat_list,
#     args={},
#     count_class=CountingStatistics,
#     **kw,
# ):
#     with ExceptionLogging(
#         "scbamtools.quant.bundle_processor", Qerr=Qerr, exc_flag=abort_flag
#     ) as el:
#         # load detailed counter configurations for each reference name
#         # plus a default setting ('*' key)
#         conf_d, channels = get_config_for_refs(args)

#         # these objects can be (re-)used by successive counter_class instances
#         stats = count_class()  # statistics on disambiguation and counting
#         uniq = set()  # keep track of (CB, UMI) tuples we already encountered

#         last_ref = None
#         counter = None
#         ref_stats = None
#         n_chunk = 0
#         count_data_chunk = []

#         bam_src = AlignedSegmentsFromQueue(Qin, abort_flag)
#         for ref, cell, umi, bundle in bam_iter_bundles(
#             bam_src, el.logger, args, stats=stats
#         ):
#             if ref != last_ref:
#                 # create a new counter-class instance with the configuration
#                 # for this reference name (genome, miRNA, rRNA, ...)
#                 if count_data_chunk:
#                     Qout.put((n_chunk, last_ref, count_data_chunk))
#                     n_chunk += 1
#                     count_data_chunk = []

#                 last_ref = ref
#                 ref_stats = stats.stats_by_ref[ref]
#                 config = conf_d.get(ref, conf_d["*"])
#                 el.logger.info(
#                     f"processing alignments to reference '{ref}'. Building counter with '{config['name']}' rules"
#                 )
#                 counter = config["counter_class"](stats=ref_stats, uniq=uniq, **config)

#             # try to select alignment and gene and determine into which
#             # channels we want to count
#             gene, channels = counter.process_bam_bundle(cell, umi, bundle)
#             if gene:
#                 count_data_chunk.append((gene, cell, channels))
#                 # print(f"add_read gene={gene} cell={cell} channels={channels}")
#                 # dge.add_read(gene=gene, cell=cell, channels=channels)

#             if len(count_data_chunk) >= args.buffer_size:
#                 Qout.put((n_chunk, ref, count_data_chunk))
#                 n_chunk += 1
#                 count_data_chunk = []

#         if count_data_chunk:
#             Qout.put((n_chunk, ref, count_data_chunk))
#             n_chunk += 1
#             count_data_chunk = []

#         el.logger.debug("done. syncing stats and gene_source")
#         stat_list.append(stats.stats_by_ref)

#         el.logger.debug("shutting down")


def DGE_counter(Qin, Qerr, abort_flag, stat_list, args={}, **kw):
    with ExceptionLogging(
        "scbamtools.quant.DGE_counter", Qerr=Qerr, exc_flag=abort_flag
    ) as el:
        # load detailed counter configurations for each reference name
        # plus a default setting ('*' key)
        conf_d, channels = get_config_for_refs(args)

        # prepare the sparse-matrix data collection (for multiple channels in parallel)
        dge = DGE(channels=channels, cell_bc_allowlist=args.cell_bc_allowlist)

        # keep track of which reference contained which gene names
        gene_source = {}

        # iterate over chunks of count_data prepared by the worker processes
        for n_chunk, ref, count_data in queue_iter(Qin, abort_flag):
            for gene, cell, channels in count_data:
                dge.add_read(gene=gene, cell=cell, channels=channels)
                # keep track of the mapping reference origin of every gene
                # (in case we combine multiple BAMs)
                if gene.startswith("mm_") and ref != "genome":
                    print(f"gene {gene} is in {ref} ? RLY?")

                gene_source[gene] = ref

            Qin.task_done()

        el.logger.debug("completed counting. Writing output.")

        sparse_d, obs, var = dge.make_sparse_arrays()
        adata = dge.sparse_arrays_to_adata(sparse_d, obs, var)
        adata.var["reference"] = [gene_source[gene] for gene in var]
        adata.uns["sample_name"] = args.sample
        adata.uns["DGE_info"] = (
            f"created with scbamtools.quant version={__version__} on {datetime.datetime.today().isoformat()}"
        )
        adata.uns["DGE_cmdline"] = sys.argv
        adata.write(args.out_dge.format(args=args))

        ## write out marginal counts across both axes:
        #   summing over obs/cells -> pseudo bulk
        #   summing over genes -> UMI/read distribution
        pname = args.out_bulk.format(args=args)

        data = OrderedDict()

        data["sample_name"] = args.sample
        data["reference"] = [gene_source[gene] for gene in var]
        data["gene"] = var
        for channel, M in sparse_d.items():
            data[channel] = sparse_summation(M)

        import pandas as pd

        df = pd.DataFrame(data).sort_values("counts")
        df.to_csv(pname, sep="\t")

        tname = args.out_summary.format(args=args)

        data = OrderedDict()

        # TODO: add back in once we dropped the Rmd QC sheet scripts
        # data["sample_name"] = args.sample
        data["cell_bc"] = obs
        for channel, M in sparse_d.items():
            data[channel] = sparse_summation(M, axis=1)

        df = pd.DataFrame(data)
        df.to_csv(tname, sep="\t")
        el.logger.debug("shutdown")


def simple_check(input, output="/dev/stdout"):
    n = 0
    for line in input:
        n += 1
    return n


def sam_to_DGE(input, channels=[], cell_bc_allow_list="", **kw):
    import scbamtools.quant

    # configs, channels = get_config_for_refs(args)

    dge = scbamtools.quant.DGE(channels=channels, cell_bc_allow_list=cell_bc_allow_list)

    counter = scbamtools.quant.SLAC_miRNACounter()
    for bundle in counter.bundles_from_SAM(input):
        cell, gene, channels = counter.process_bundle(bundle)
        if cell != "-":
            dge.add_read(cell, gene, channels)

    return dge.make_sparse_arrays()


def main(args):
    configs, channels = get_config_for_refs(args)
    # from pprint import pprint

    # pprint(configs)

    # from scbamtools.parallel import parallel_BAM_workflow
    util.ensure_path(args.output + "/")

    logger = util.setup_logging(args, f"scbamtools.quant.main")
    logger.info("startup")

    config = configs["*"]
    w = (
        mf.Workflow("count", total_pipe_buffer_MB=args.pipe_buffer_size)
        .BAM_reader(
            input=args.bam_in[0],
            threads=args.input_threads,
            output=mf.FIFO("input_sam", "w"),
        )
        .distribute(
            func=mf.parts.distribute_by_CB,
            input=mf.FIFO("input_sam", "r"),
            outputs=mf.FIFO("dist_{n}", "w", n=args.worker_threads),
            header_detect_func=mf.util.is_header,
            header_fifo="/dev/null",
        )
        .workers(
            func=sam_to_DGE,
            input=mf.FIFO("dist_{n}", "r"),
            channels=channels,
            cell_bc_allow_list=args.cell_bc_allow_list,
            n=args.worker_threads,
        )
        .run()
    )

    from scbamtools.quant import DGE

    adata_shards = []
    for name, res in w.result_dict.items():
        print(name, "retrieving adata object")
        if name.startswith("count.worker"):
            channel_d, obs_names, var_names = res
            adata = DGE.sparse_arrays_to_adata(channel_d, obs_names, var_names)
            adata_shards.append(adata)

    import anndata

    adata = anndata.concat(
        adata_shards, axis=0, join="outer", merge="unique", uns_merge="first"
    )
    # number of molecules counted for this cell
    adata.obs[f"n_counts"] = sparse_summation(adata.X, axis=1)
    # number of genes detected in this cell
    adata.obs["n_genes"] = sparse_summation(adata.X > 0, axis=1)
    for channel in adata.layers.keys():
        adata.obs[f"n_{channel}"] = sparse_summation(adata.layers[channel], axis=1)

    adata.write_h5ad(args.dge_out)
    return adata

    # if not stats:
    #     return -1

    # if args.out_stats:
    #     stats.save_stats(args.out_stats.format(**locals()))


if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
