from fixtures import *


def test_help():
    try:
        ubam("--help")
    except SystemExit:
        pass


def test_dropseq():
    ubam(
        "--read1",
        scbamtools_dir + "test_data/adap_test.R1.fastq",
        "--read2",
        scbamtools_dir + "test_data/adap_test.R2.fastq.gz",
        "--bam-out",
        "/dev/null",
        # "--flavor", "dropseq",
    )


def test_single():
    ubam(
        "--read2",
        scbamtools_dir + "test_data/adap_test.R2.fastq.gz",
        "--bam-out",
        "/dev/null",
        """--cell='"ACGTACGTACGTACGT"'""",
    )


def test_minqual():
    ubam(
        "--read2",
        scbamtools_dir + "test_data/adap_test.R2.fastq.gz",
        "--bam-out",
        "/dev/null",
        "--min-qual",
        "30",
        """--cell='"ACGTACGTACGTACGT"'""",
    )
