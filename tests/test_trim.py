from fixtures import *


def test_help():
    try:
        trim("--help")
    except SystemExit:
        pass


def test_dropseq():
    trim(
        scbamtools_dir + "test_data/adap_test.bam",
        "--bam-out",
        "test_data/adap_test.out.bam",
        "--flavor",
        "dropseq",
    )
