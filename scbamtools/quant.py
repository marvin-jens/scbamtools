import numpy as np
import logging
from collections import defaultdict, OrderedDict
from scbamtools.contrib import __version__, __author__, __license__, __email__

# from mrfifo.util import CountingStatistics
import scbamtools.util as util

"""
This module implements the functionality needed to aggregate annotated alignments from a BAM file
into a digital gene expression (DGE) matrix.
The matrix has gene names as columns and cell barcodes as rows. It is stored as a scipy.sparse.csr_matrix
to save RAM. Depending on the kinds of features a read can be annotated with, counting goes through
two stages of disambiguation (see below) and then a determination of "how to count" the read. A read can be 
counted into more than one count-matrix which will be stored as 'layers' in the resulting AnnData object
(see below).

Configuration of the counting rules is done via the config.yaml. All parameters below have a 
corresponding entry there. The section "quant" holds is subdivided by flavors (for example "default") 
and each flavor can set the values of any of the options below. If you choose to customize further,
you can even set a different (sub-)class to be used for counting and potentially define additional paramters
to configure it in the YAML as well.

## BAM ingestion

BAM records are first grouped into 'bundles' of alternative alignments for the same read. Each bundle
consists of a cell barcode, a UMI sequence and a list of one or more possible alignments. For the purpose
of counting gene expression, the different alignments are represented by the values of the gn, gf, and gt 
BAM tags.

## alignment disambiguation

If we have more than one alignment in a bundle, the next step is to try and disambiguate between them.
For this purpose, the alignments are assigned priorites from the 'alignment_priorities' dictionary.
If there is a tie with multiple alignments having the same priority, the disambiguation failed and
the read can not be counted (for example aligning to two paralogous genes with exactly the same alignment
score). However, if there is one alignment with the highest priority, disambiguation has succeeded 
(for example an exonic region of one specific gene being preferred over alternative intergenic 
alignments).

## gene disambiguation

Even if a single alignment can be selected, there can still be conflicting annotations for a read: it 
may align to a region that is annotated for more than one overlapping genes. In this case another round
of disambiguation is attempted, this time using the 'gene_priorities' dictionary.
Here, for example, a coding exon annotation in the sense direction of the read would take precedence
over an intronic annotation to another gene on the opposite strand.

## channel determination

Lastly, if a read has been found to be assignable to one gene, we may still count it into different
channels (note, within each channel, the sum of all counts will never exceed the number of reads
Reads are never counted multiple times into the same channel. That includes the main adata.X channel).

For example, if a read aligns in an intronic region, it can be counted into 'intronic_reads'. If it is
furthermore the first time we observe this combination of cell barcode and UMI, it would additionally be 
counted as 'intronic_counts' ('counts' meaning "unique molecules").
If 'intronic_counts' are included in the "default_X_counts" parameters, it would also count to the main
DGE matrix stored as adata.X .

## A note on parallelization

In order to decide if a read has been seen more than once already (copies of the same BC & UMI sequences)
each DefaultCounter instance maintains a set() of hash values. To ensure correctness but still allow
parallelization, the input process will place bundles into distinct queues based on the cell barcode
sequence.

For example, if we want to run 4 counting-processes in parallel, we would map the first nucleotide to the 
processes such that barcodes starting with 'A'-> process 1, 'C' -> process 2, 'G' -> process 3, 
'T' -> process 3. In this manner the separate sets of hash values are guaranteed to never overlap and the
test for UMI uniqueness is not an issue.
"""

default_alignment_priorities = {
    "C": 100,  # coding exon
    "U": 100,  # UTR exon
    "N": 90,  # exon of non-coding transcript
    "I": 50,  # intronic region
    # lower case == antisense
    "c": 10,
    "u": 10,
    "n": 9,
    "i": 5,
    # intergenic
    "-": 0,
}
default_gene_priorities = {
    "C": 110,  # coding exon
    "U": 100,  # UTR exon
    "N": 90,  # exon of non-coding transcript
    "I": 50,  # intronic region
    "c": 11,
    "u": 10,
    "n": 9,
    "i": 5,
    "-": 0,
}
default_exonic_tags = ["C", "U", "N"]
default_exonic_tags += [t.lower() for t in default_exonic_tags]
default_intronic_tags = ["I", "i"]

default_channels = [
    "counts",
    "reads",
    "exonic_counts",
    "exonic_reads",
    "intronic_counts",
    "intronic_reads",
]
default_X_counts = [
    "exonic_counts",
    "intronic_counts",
]  # what should be counted into adata.X matrix
default_X_reads = [
    "exonic_reads",
    "intronic_reads",
]  # what should be counted into adata.layers["reads"] matrix (correspond with adata.X but for reads not UMIs)


class BaseCounter:

    logger = logging.getLogger("scbamtools.quant.BaseCounter")

    def __init__(
        self,
        channels=default_channels,
        exonic_tags=default_exonic_tags,
        intronic_tags=default_intronic_tags,
        X_counts=default_X_counts,
        X_reads=default_X_reads,
        handle_multimappers=True,
        uniq=set(),
        stats=defaultdict(int),
        **kw,
    ):

        self.kw = kw
        self.filter_sam_records = None  # can be set to a callable which may perform arbitrary filter operations on the SAM columns
        self.channels = channels
        self.exonic_set = set(exonic_tags)
        self.intronic_set = set(intronic_tags)

        # which channels shall contribute to the adata.X (main channel)
        self.count_X_channels = set(X_counts)
        # which channels shall contribute to the adata.layers['reads'] (main channel reads version)
        self.read_X_channels = set(X_reads)
        self.handle_multimappers = handle_multimappers

        self.stats = stats
        self.uniq = uniq

    def bundles_from_SAM(self, sam_input):
        """
        Generator which processes an input text stream of SAM-formatted alignment records.
        Consecutive records for the same read are grouped and emitted together as a bundle,
        which is a list of (chrom, strand, tags).
        tags at this level is a dictionary with strings or integers. gn,gf are not yet split on
        ','.
        """

        default_tags = {"gn": "-", "gf": "-", "AS": -1, "CB": "-", "MI": ""}

        def extract(cols):
            # yield chrom, strand, gn, gf, score
            chrom = cols[2]
            strand = "-" if int(cols[3]) & 16 else "+"

            tags = default_tags.copy()
            for c in cols[11:]:
                # AA:Z:BCD -> tags['AA'] = 'BCD'
                tname, ttype, tvalue = c.split(":", maxsplit=2)
                if ttype == "i":
                    tags[tname] = int(tvalue)
                else:
                    tags[tname] = tvalue

            return chrom, strand, tags

        bundle = []
        last_qname = ""
        for sam in sam_input:
            self.stats["SAM_records_total"] += 1
            cols = sam.rstrip().split("\t")
            if self.filter_sam_records:
                if not self.filter_sam_records(cols):
                    continue

            qname = cols[0]
            if qname != last_qname:
                if bundle:
                    yield bundle

                bundle = []
                last_qname = qname
            bundle.append(extract(cols))

        if bundle:
            yield bundle

    ## Implement/overload the following functions in a sub-class to implement a desired counting behavior
    def unique_alignment(self, chrom, strand, tags):
        """
        Gets called by select_alignment() if we have exactly one record of (chrom, strand, tags) for an aligned read.
        May be overloaded to provide a fast-path.
        """
        self.stats["N_aln_unique"] += 1

        # nothing to do here, we have a unique alignment
        return chrom, strand, tags

    def select_multimapper(self, bundle):
        self.stats["N_aln_multi"] += 1
        self.stats["N_aln_selection_failed"] += 1

        return None

    def select_alignment(self, bundle):
        """
        Gets called to select one record from a bundle of [(chrom, strand, tags),] for an aligned read.
        May be overloaded to provide selection by best alignment-score or preference of genic over
        intergenic alignments (see mRNACounter).
        Should return a single (chrom, strand, tags) tuple or None
        Base class returns None for multimappers which results in multimappers not being counted at all.
        """
        if len(bundle) == 1:
            selected = self.unique_alignment(*bundle[0])
        else:
            selected = self.select_multimapper(bundle)

        return selected

    def select_gene(self, chrom, strand, tags):
        """
        Gets called after an alignment has been selected (either unique or by preference) in order to extract
        gene information from the BAM tags and return the name of the gene that should be counted, along with
        the functional category of how the alignment was classified.
        Can implement more complex rules such as preference of exonic and sense gene hits over
        intronic/antisense gene hits (see mRNACounter).
        Default implementation returns first (gene, [function,]) annotation if all annotations
        are for the same gene. Returns ('-', '-') for missing information, and None for multiple genes
        (which then means the alignment is discarded).
        """
        gn = tags.get("gn", "-").split(",")
        gf = tags.get("gf", "-").split(",")

        if len(gn) == 1:
            if gn[0] == "-":
                self.stats["N_no_gene"]
            else:
                self.stats["N_gene_unique"] += 1
            selected = gn[0], [gf[0]]
        elif len(set(gn)) == 1:
            self.stats["N_gene_unique"] += 1
            gene = gn[0]
            selected = gene, [f for f, g in zip(gf, gn) if g == gene]
        else:
            self.stats["N_gene_multi"] += 1
            self.stats["N_gene_selection_failed"] += 1

            selected = None

        return selected

    def exon_intron_disambiguation(self, channels):
        # how to handle reads that align to both intron and exon features
        # default implementation counts everything independently
        return channels

    ## Count-channel determination
    def select_channels(self, gf, uniq, tags):
        channels = set()
        exon = False
        intron = False

        if gf[0].islower():
            self.stats["N_aln_antisense"] += 1
        else:
            self.stats["N_aln_sense"] += 1

        for f in gf:
            if f in self.exonic_set:
                channels.add("exonic_reads")
                exon = True
                if uniq:
                    channels.add("exonic_counts")

            elif f in self.intronic_set:
                channels.add("intronic_reads")
                intron = True
                if uniq:
                    channels.add("intronic_counts")

        # post-process: if both exon & intron annotations
        # (but from different isoforms) are present
        # decide how to count
        if exon and intron:
            channels = self.exon_intron_disambiguation(channels)

        if channels & self.count_X_channels:
            channels.add("counts")

        if channels & self.read_X_channels:
            channels.add("reads")

        for c in channels:
            self.stats[f"N_channel_{c}"] += 1

        if not channels:
            self.stats[f"N_channel_NONE"] += 1

        return channels

    def process_bundle(self, bundle):
        """
        main function: alignment bundle -> counting channels.
        This is where the logic implemented by select_alignment(), select_gene() etc is wired together.

        bundle->[if non-unique->select_alignment()]->select_gene()->determine_gene_and_channels()
        """
        self.stats["SAM_bundles_total"] += 1
        gene = None
        selected = None
        channels = set()

        selected = self.select_alignment(bundle)
        if selected:
            chrom, strand, tags = selected
            gene, gf = self.select_gene(chrom, strand, tags)
            if (gene is not None) and (gene != "-"):
                # TODO: discuss merits of counting intergenic. It may be a
                # meaningful QC stat for DNA contamination. For now: discard

                # print(f"gene={gene} gf={gf}")

                # handle the whole uniqueness in a way that can parallelize
                # maybe split by UMI[:k] or CB[:k]? This way, distributed uniq() sets would
                # never overlap
                cell = tags["CB"]
                umi = tags["MI"]
                key = hash((cell, umi))
                uniq = not (key in self.uniq)
                if uniq:
                    self.uniq.add(key)

                channels = self.select_channels(gf, uniq, tags)
                return cell, gene, channels

        # print(f"process_bundle()-> cell={cell} gene={gene}, channels={channels}")


class CustomIndexCounter(BaseCounter):

    logger = logging.getLogger("scbamtools.quant.CustomIndexCounter")

    def unique_alignment(self, chrom, strand, tags):
        tags["gn"] = [chrom]
        tags["gf"] = ["N"]
        return (chrom, strand, tags)

    def select_alignment(self, bundle):
        # pick the first alignment on the + strand
        # this *should* also be the alignment with highest score
        # unless there are ties, in which case we just pick the
        # first hit
        for chrom, strand, tags in bundle:
            if strand == "+":
                return self.unique_alignment(chrom, strand, tags)

    def select_channels(self, gf, uniq, tags):
        if uniq:
            # channels = {"reads", chan}
            channels = {"reads", "counts"}
        else:
            channels = {"reads"}

        return channels


class miRNAPrecursorCounter(CustomIndexCounter):

    logger = logging.getLogger("scbamtools.quant.miRNAPrecursorCounter")

    def select_alignment(self, bundle):
        # has to return either (chrom, strand, tags) or None
        chrom, strand, tags = bundle[0]
        # return the first alignment, no matter how many alternatives we find
        return (chrom, strand, tags)

    def select_gene(self, chrom, strand, tags):
        # give precedent to loop
        gn = tags["gn"].split(",")
        gf = tags["gf"].split(",")
        for g, f in zip(gn, gf):
            if g.endswith("loop"):
                return g, f
        else:
            return gn[0], gf[0]

    def select_channels(self, gf, uniq, tags):
        channels = super().select_channels(gf, uniq, tags)
        if uniq and ("polyA" in tags.get("A3", "")):
            channels.add("polyA")


class mRNACounter(BaseCounter):

    logger = logging.getLogger("scbamtools.quant.mRNACounter")

    def __init__(
        self,
        alignment_priorities=default_alignment_priorities,
        gene_priorities=default_gene_priorities,
        **kw,
    ):

        BaseCounter.__init__(self, **kw)

        self.alignment_priorities = alignment_priorities
        self.gene_priorities = gene_priorities

    def unique_alignment(self, chrom, strand, tags):
        # no processing needed here. We have a unique alignment
        return chrom, strand, tags

    ## Try to select the most meaningful (cDNA-derived) from multiple
    ## reported alignments for a read
    def select_alignment(self, bundle):
        """
        If multiple alignments are reported for one fragment/read try to make a reasonable choice:
        If one of the alignments is to a coding exon, but the others are not, choose the exonic one.
        If multiple alternative alignments point to exons of different coding genes, do not count anything
        because we can not disambiguate.
        """
        from collections import defaultdict

        def get_prio(gf):
            m_prio = -1
            for x in gf.replace(",", "|").split("|"):
                prio = self.gene_priorities.get(x, 0)
                if prio > m_prio:
                    m_prio = prio

            return m_prio

        top_prio = 0
        n_top = 0
        kept = None

        for chrom, strand, tags in bundle:
            gf = tags["gf"]

            p = get_prio(gf)
            if p > top_prio:
                top_prio = p
                kept = (chrom, strand, tags)
                n_top = 1
            elif p == top_prio:
                n_top += 1
                # kept = (chrom, strand, tags)

        if n_top == 1:
            # print(kept)
            return kept

    ## Gene selection strategy, similar to alignment selection
    def select_gene(self, chrom, strand, tags):
        # let's see if we can prioritize which gene we are interested in
        gene_prio = defaultdict(int)
        gene_gf = defaultdict(set)
        max_prio = 0
        max_code = "-"

        gn = tags["gn"].split(",")
        gf = tags["gf"].split(",")

        import itertools

        for n, f in zip(gn, gf):
            codes = f.split("|")
            f_prios = np.array([self.gene_priorities.get(x, 0) for x in codes])
            i = f_prios.argmax()
            p = f_prios[i]
            # keep only the highest priority codes per compound code,
            # examples:
            # C|I -> C
            # N|U|I -> U
            c = codes[i]
            gene_gf[n].add(c)

            if p < 0:
                gene_prio[n] -= p
                # we get a penalty. Useful if the overlap extends
                # beyond the boundaries of a feature
            elif p >= gene_prio[n]:
                # we have found a higher or tied priority
                gene_prio[n] = p
                max_prio = max([max_prio, p])

        # restrict to genes tied for highest priority hits
        gn_new = [n for n in set(gn) if gene_prio[n] == max_prio]
        # print(len(gn_new))
        if len(gn_new) == 1:
            # YES we can salvage this read
            gene = gn_new[0]
            res = gene, sorted(gene_gf[gene])
        else:
            # NO still ambiguous
            res = "-", "-"

        # print(f"gn={gn} gf={gf} -> res={res}")
        return res

    def exon_intron_disambiguation(self, channels):
        return channels - set(["exonic_reads", "exonic_counts"])


class SLAC_miRNACounter(BaseCounter):

    logger = logging.getLogger("scbamtools.quant.SLAC_miRNACounter")

    def __init__(
        self,
        *argc,
        max_5p_soft_clip=5,
        max_3p_soft_clip=None,
        min_matches=15,
        max_pos=3,
        **kwargs,
    ):
        super().__init__(self, *argc, **kwargs)
        self.max_5p_soft_clip = max_5p_soft_clip
        self.max_3p_soft_clip = max_3p_soft_clip
        self.min_matches = min_matches
        self.max_pos = max_pos
        self.filter_sam_records = self.cigar_filter

    def cigar_filter(self, cols):
        import re

        if self.max_pos is not None:
            pos = int(cols[3])
            if pos > self.max_pos:
                return False

        matches = 0
        cigar = cols[5]
        cigar_ops = re.findall(r"(\d+)(\w)", cigar)

        # check 5' most operation
        n, o = cigar_ops.pop(0)
        n = int(n)
        if o == "S":
            if n > self.max_5p_soft_clip:
                self.stats["CIGAR_excessive_5p_clipping"] += 1
                return False
        elif o == "M":
            matches += n

        if cigar_ops and (self.max_3p_soft_clip is not None):
            # check 3' most operation
            n, o = cigar_ops.pop(-1)
            n = int(n)
            if o == "S":
                if n > self.max_3p_soft_clip:
                    self.stats["CIGAR_excessive_3p_clipping"] += 1
                    return False
            elif o == "M":
                # assume it is 'M'
                matches += n

        if self.min_matches is not None:
            if cigar_ops:
                for n, o in cigar_ops:
                    if o == "M":
                        matches += int(n)

            if matches <= self.min_matches:
                self.stats["CIGAR_insufficient_match"] += 1
                return False

        self.stats["CIGAR_pass"] += 1
        return True

    def unique_alignment(self, bundle):
        CB, MI, chrom, strand, _, _, score = bundle[0]
        return (CB, MI, chrom, strand, [chrom], ["N"], score)

    def select_alignment(self, bundle):
        # only consider alignments on the + strand (for custom indices)
        best_score = 0
        best_score_aln = None
        best_score_tie = 0

        for aln in bundle:
            CB, MI, chrom, strand, gn, gf, score = aln
            if strand == "+":
                if score > best_score:
                    best_score = score
                    best_score_aln = CB, MI, chrom, strand, [chrom], ["N"], score
                    best_score_tie = 1
                elif score == best_score:
                    best_score_tie += 1

        if best_score_tie == 1:
            return best_score_aln

    def select_gene(self, CB, MI, chrom, strand, gn, gf, score):
        # print(CB, MI, chrom, gn, gf)
        return [chrom], ["N"]


DefaultCounter = mRNACounter


def sparse_summation(X, axis=0):
    # for csr_array this would be fine
    # return np.array(X.sum(axis=axis))
    # unfortunately, for the time being scanpy needs csr_matrix
    # due to h5py lagging somewhat
    if axis == 0:
        return np.array(X.sum(axis=0))[0]
    elif axis == 1:
        return np.array(X.sum(axis=1))[:, 0]


class DGE:
    logger = logging.getLogger("scbamtools.quant.DGE")

    def __init__(self, channels=["count"], cell_bc_allow_list=""):
        self.channels = channels
        self.counts = defaultdict(lambda: defaultdict(int))
        self.DGE_cells = set()
        self.DGE_genes = set()
        self.allowed_bcs = self.load_list(cell_bc_allow_list)

    @staticmethod
    def load_list(fname):
        if fname:
            allowed_bcs = set([line.rstrip() for line in open(fname)])
            DGE.logger.debug(f"restricting to {len(allowed_bcs)} allowed cell barcodes")
            return allowed_bcs
        else:
            DGE.logger.debug("all barcodes allowed")

    def add_read(self, cell, gene, channels=["count"]):
        if channels:
            if self.allowed_bcs and not (cell in self.allowed_bcs):
                # print(f"discarding '{cell}' as NA", cell in self.allowed_bcs)
                cell = "NA"

            self.DGE_cells.add(cell)
            self.DGE_genes.add(gene)

            d = self.counts[(gene, cell)]
            for channel in channels:
                d[channel] += 1

    def make_sparse_arrays(self):
        """_summary_
        Convert counts across all channels into a dictionary of sparse arrays

        Returns:
            dict: channel is key, value is scipt.sparse.csr_array
        """
        import scipy.sparse
        import anndata

        ind_of_cell = {}
        obs = []
        for cell in sorted(self.DGE_cells):
            ind_of_cell[cell] = len(obs)
            obs.append(cell)

        ind_of_gene = {}
        var = []
        for gene in sorted(self.DGE_genes):
            ind_of_gene[gene] = len(var)
            var.append(gene)

        counts_by_channel = defaultdict(list)
        row_ind = []
        col_ind = []
        for (gene, cell), channel_dict in self.counts.items():
            for channel in self.channels:
                counts_by_channel[channel].append(channel_dict[channel])

            row_ind.append(ind_of_cell[cell])
            col_ind.append(ind_of_gene[gene])

        if (not len(row_ind)) or (not len(col_ind)):
            self.logger.warning(
                f"empty indices len(row_ind)={len(row_ind)} len(col_ind)={len(col_ind)}"
            )
            return {}, [], []

        sparse_channels = OrderedDict()
        for channel in self.channels:
            counts = counts_by_channel[channel]
            self.logger.debug(
                f"constructing sparse-matrix for channel '{channel}' from n={len(counts)} non-zero entries (sum={np.array(counts).sum()})"
            )
            m = scipy.sparse.csr_matrix((counts, (row_ind, col_ind)), dtype=np.float32)
            self.logger.debug(
                f"resulting sparse matrix shape={m.shape} len(obs)={len(obs)} len(var)={len(var)}"
            )
            sparse_channels[channel] = m

        return sparse_channels, obs, var

    @staticmethod
    def sparse_arrays_to_adata(channel_d, obs, var, main_channel="counts"):
        """_summary_
        Creates an AnnData object from sparse matrices in the channel_d dictionary. The main_channel
        will be assigned to adata.X, the others to adata.layers[channel].

            obs (cell barcodes) -> rows
            var (gene names) -> columns

        Args:
            channel_d (_type_): _description_
            obs (_type_): _description_
            var (_type_): _description_
            main_channel (_type_): _description_

        Returns:
            AnnData object: a proper, scanpy-compatible AnnData object with (optional) layers.
        """
        import anndata

        # print("channel_d", channel_d.keys())
        X = channel_d[main_channel]
        adata = anndata.AnnData(X)
        adata.obs_names = obs
        adata.var_names = var

        for channel, sparse in channel_d.items():
            if channel != main_channel:
                adata.layers[channel] = sparse
                # add marginal counts
                adata.obs[f"n_{channel}"] = sparse_summation(
                    adata.layers[channel], axis=1
                )

        # number of molecules counted for this cell
        adata.obs[f"n_counts"] = sparse_summation(adata.X, axis=1)
        # number of genes detected in this cell
        adata.obs["n_genes"] = sparse_summation(adata.X > 0, axis=1)
        return adata
