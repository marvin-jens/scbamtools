import pytest
import os
import sys

scbamtools_dir = os.path.dirname(__file__) + "/../"


@pytest.fixture(scope="session")
def test_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("root")

    return tmp


def run(cmd, *argc, expect_fail=False):
    sys.argv = [
        f"{cmd}.py",
    ] + list(argc)
    from importlib import import_module

    mod = import_module(f"scbamtools.bin.{cmd}")
    res = mod.cmdline()
    print("got result", res)

    if expect_fail:
        assert isinstance(res, Exception) == True
    else:
        assert isinstance(res, Exception) == False


def ubam(*argc, **kwargs):
    run("uBAM", *argc, **kwargs)


def trim(*argc, **kwargs):
    run("trim", *argc, **kwargs)


def ann(*argc, **kwargs):
    run("ann", *argc, **kwargs)
