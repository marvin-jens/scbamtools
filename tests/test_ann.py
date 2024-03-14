import pytest
import os
from scbamtools.annotation import GenomeAnnotation, CompiledClassifier

@pytest.fixture(scope="session")
def test_compile(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("simple")
    scbamtools_path = os.path.dirname(__file__)
    ga = GenomeAnnotation.from_GTF(
        os.path.join(scbamtools_path, "../test_data/simple_annotation.gtf"),
        df_cache=(tmp / "simple.tsv").as_posix()
    )
    compiled_path = (tmp / "compiled/").as_posix()
    _ = ga.compile(compiled_path)
    
    return tmp / "compiled"

def test_recompile(test_compile):
    assert CompiledClassifier.files_exist(test_compile)
    ga = GenomeAnnotation.from_compiled_index(test_compile)
    try:
        ga.compile()
    except ValueError:
        pass
    else:
        raise ValueError("attempt to re-compile already compiled annotation should have failed!")
    
    ga2 = GenomeAnnotation.from_uncompiled_df((test_compile / "../simple.tsv").as_posix()).compile(test_compile.as_posix())
    assert (ga2.classifier.classifications == ga.classifier.classifications).all()


def test_regions(test_compile):
    """
    chr1    test    transcript      1       10000   .       +       .       gene_name "A"; transcript_type "protein_coding"; transcript_id "A.1";
    chr1    test    exon    1       150     .       +       .       gene_name "A"; transcript_type "protein_coding"; transcript_id "A.1";
    chr1    test    exon    9000    10000   .       +       .       gene_name "A"; transcript_type "protein_coding"; transcript_id "A.1";
    chr1    test    CDS     75      150     .       +       .       gene_name "A"; transcript_type "protein_coding"; transcript_id "A.1";
    chr1    test    CDS     9000    9050    .       +       .       gene_name "A"; transcript_type "protein_coding"; transcript_id "A.1";

    chr2    test    transcript      1       10000   .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.1";
    chr2    test    exon    1       150     .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.1";
    chr2    test    exon    9000    10000   .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.1";
    chr2    test    transcript      1       10000   .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";
    chr2    test    exon    1       150     .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";
    chr2    test    exon    500     600     .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";
    chr2    test    exon    9000    10000   .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";
    chr2    test    CDS     75      150     .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";
    chr2    test    CDS     500     600     .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";
    chr2    test    CDS     9000    9050    .       +       .       gene_name "B"; transcript_type "protein_coding"; transcript_id "B.2";

    chr3    test    transcript      1       10000   .       +       .       gene_name "C"; transcript_type "protein_coding"; transcript_id "C.1";
    chr3    test    exon    1       150     .       +       .       gene_name "C"; transcript_type "protein_coding"; transcript_id "C.1";
    chr3    test    exon    9000    10000   .       +       .       gene_name "C"; transcript_type "protein_coding"; transcript_id "C.1";
    chr3    test    CDS     75      150     .       +       .       gene_name "C"; transcript_type "protein_coding"; transcript_id "C.2";
    chr3    test    CDS     9000    9050    .       +       .       gene_name "C"; transcript_type "protein_coding"; transcript_id "C.2";
    chr3    test    transcript      1000    2100    .       -       .       gene_name "D"; transcript_type "lincRNA"; transcript_id "D.1";                                                                     
    chr3    test    exon    1000    2100    .       -       .       gene_name "D"; transcript_type "lincRNA"; transcript_id "D.1";
   
    """
    regions = [
        # intergenic
        (('chr1', '+', [(20000, 20010), ]), ("-", "-", "-")),
        (('chrA', '+', [(20000, 20010), ]), ("-", "-", "-")),
        # single gene
        (('chr1', '+', [(0, 74), ]), ("A", "N", "C")),
        (('chr1', '+', [(75, 150), ]), ("A", "C", "C")),
        (('chr1', '+', [(0, 150), ]), ("A", "N,C", "C")),
        (('chr1', '+', [(75, 151), ]), ("A", "C,I", "C")),
        (('chr1', '-', [(0, 150), ]), ("A", "n,c", "C")),
        (('chr1', '+', [(8050, 9040), ]), ("A", "I,C", "C")),
        (('chr1', '+', [(8090, 9051), ]), ("A", "I,C,N", "C")),
        # alternative splicing
        (('chr2', '+', [(450, 550), ]), ("B", "I,C|I", "C")),
        (('chr2', '+', [(140, 150), (500,590)]), ("B", "C|N,C|I", "C")),
        (('chr2', '+', [(9000, 10000)]), ("B", "C|N,N", "C")),
        # multiple genes on opposite strands
        # TODO: we may want to use the information if part of a read aligns 
        # outside of an annotated feature to de-prioritize it
        (('chr3', '+', [(950, 1050), ]), ("C,D", "I,n", "C,I")),
        (('chr3', '+', [(900, 1000), ]), ("C,D", "I,n", "C,I")),
        (('chr3', '-', [(900, 1000), ]), ("C,D", "i,N", "C,I")),
        (('chr3', '*', [(900, 1000), ]), ("C,D", "I,N", "C,I")),

    ]

    ga = GenomeAnnotation.from_compiled_index(test_compile)
    
    fail = []
    for (chrom, strand, blocks), expect in regions:
        res = ga.get_annotation_tags(chrom, strand, blocks)
        gn, gf, gt = res
        print(f'gn "{gn}"; gf "{gf}"; gt "{gt}"; expect={expect}')
        if res != expect:
            fail.append((chrom, strand, blocks, res, expect))
    
    for f in fail:
        print(f"failed test: {f}")

    assert len(fail) == 0


