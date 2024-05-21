import pytest
import scbamtools.quant as quant
import os

test_bundles = [
    ("empty", [], ("-", "-", set())),
    (
        "uniq",
        [("ACGTACGT", "AAAAC", "chr1", "+", "A", "N", 0)],
        ("A", {"counts", "reads", "exonic_counts", "exonic_reads"}),
    ),
    (
        "mm_gene_intergenic",
        [
            ("ACGTACGT", "AAAAG", "chr1", "+", "A", "C", 0),
            ("ACGTACGT", "AAAAG", "chr2", "-", "-", "-", 0),
        ],
        ("ACGTACGT", "A", {"counts", "reads", "exonic_counts", "exonic_reads"}),
    ),
    (
        "mm_geneC_geneI",
        [
            ("ACGTACGT", "AAACA", "chr1", "+", "A", "C", 0),
            ("ACGTACGT", "AAACA", "chr2", "-", "B", "I", 0),
        ],
        ("ACGTACGT", "A", {"counts", "reads", "exonic_counts", "exonic_reads"}),
    ),
    (
        "mm_complex",
        [
            ("ACGTACGT", "ACACA", "chr1", "+", "A", "C|N|I,I", 0),
            ("ACGTACGT", "ACACA", "chr2", "-", "B", "n", 0),
        ],
        ("ACGTACGT", "A", {"counts", "reads", "intronic_counts", "intronic_reads"}),
    ),
    (
        "mm_complex2",
        [
            ("ACGTACGT", "ATACA", "chr1", "+", "A", "N,C|N|I,U", 0),
            ("ACGTACGT", "ATACA", "chr2", "-", "B", "n", 0),
        ],
        ("ACGTACGT", "A", {"counts", "reads", "exonic_counts", "exonic_reads"}),
    ),
    (
        "mm_ambig",
        [
            ("ACGTACGT", "AAACG", "chr1", "+", "A", "C", 0),
            ("ACGTACGT", "AAACG", "chr2", "-", "B", "C", 0),
        ],
        ("ACGTACGT", None, set()),
    ),
    (
        "mm_geneI_geneN",
        [
            ("ACGTACGT", "AAACT", "chr1", "+", "A,B", "I,n", 0),
        ],
        ("ACGTACGT", "A", {"counts", "reads", "intronic_counts", "intronic_reads"}),
    ),
    (
        "uniq_dup",
        [("ACGTACGT", "AAAAC", "chr1", "+", "A", "N", 0)],
        ("ACGTACGT", "A", {"reads", "exonic_reads"}),
    ),
    (
        "uniq",
        [("NNNNNNNN", "AAAAC", "chr1", "+", "A", "N", 0)],
        ("NNNNNNNN", "A", {"counts", "reads", "exonic_counts", "exonic_reads"}),
    ),
]


def test_gene_selection():
    counter = quant.DefaultCounter()
    assert counter.select_gene("NA", "NA", "chr1", "+", "A,B", "C|U|I,i|n", 90) == (
        "A",
        ["C"],
    )
    assert counter.select_gene("NA", "NA", "chr1", "+", "A", "C|I,I,U", 90) == (
        "A",
        ["C", "I", "U"],
    )


def test_default_counter():
    counter = quant.DefaultCounter()
    sm_dir = os.path.dirname(__file__)
    dge = quant.DGE(
        channels=counter.channels,
        cell_bc_allow_list=sm_dir + "/../test_data/simple_allowlist.txt",
    )

    fail = []
    for bla in test_bundles:
        assert len(bla) == 3
        name, bundle, expect = bla
        res = counter.process_bundle(bundle=bundle)
        cb, gene, channels = res
        print(res == expect, res, expect)
        if res != expect:
            fail.append((name, bundle, res, expect))

        dge.add_read(gene, cb, channels)

    channel_d, obs, var = dge.make_sparse_arrays()

    adata = quant.DGE.sparse_arrays_to_adata(channel_d, obs, var)
    print(adata)
