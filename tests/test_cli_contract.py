"""What the documentation build depends on, pinned here so it cannot quietly break.

The docs site renders the CLI reference from the real ``ArgumentParser``, which is what
stops published usage from drifting away from ``--help``. Two things have to hold for
that: the parser must be reachable without running it, and importing the module must not
require the conda-only half of the environment — otherwise the GitHub Actions job needs
micromamba and flyem-forge instead of a plain pip install. That matters most here, since
this package is the one that actually depends on vol2mesh and dvidutils.
"""

import argparse
import subprocess
import sys

CONDA_ONLY = ["vol2mesh", "dvidutils", "kimimaro", "DracoPy", "osteoid", "tensorstore",
              "h5py"]


def test_build_parser_returns_the_parser_without_running_it():
    from neu_morpho.cli import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "neu-morpho"


def test_every_subcommand_is_reachable_from_the_parser():
    from neu_morpho import cli

    parser = cli.build_parser()
    subs = next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices
    assert set(subs) == {"run", "progress", "run-report"}
    for name, sub in subs.items():
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_run_validation_still_reports_against_the_run_subparser():
    """`build_parser` hands `_parse_args` the run subparser so its errors keep naming
    `neu-morpho run ...` rather than the top-level usage."""
    from neu_morpho import cli

    parser = cli.build_parser()
    assert getattr(parser, "_run_parser", None) is not None
    assert parser._run_parser.prog.endswith("run")


def test_every_stage_is_described():
    """`STAGE_DOC` is the only description of the stages, and three places read it:
    `--help`, the published CLI reference, and the docs cheat sheet. A stage added to
    the tuple without a description would silently go undocumented in all three."""
    from neu_morpho.cli import STAGE_DOC, STAGES

    assert tuple(STAGE_DOC) == STAGES
    assert set(STAGES) == {"seg", "index", "mesh", "skel"}
    for name, doc in STAGE_DOC.items():
        assert len(doc) > 40, f"{name} has no real description"


def test_the_stage_list_reaches_run_help():
    """It lives in the description, not in `--stages`' help, because argparse's default
    formatter re-wraps argument help and collapses the block into a wall of text."""
    from neu_morpho.cli import STAGES, build_parser

    run = build_parser()._run_parser
    assert "Stages" in run.description
    for name in STAGES:
        assert name in run.description, f"{name} missing from run --help"
    # the wrapped block must survive intact, which is what RawDescription buys
    assert "\n" in run.description
    rendered = run.format_help()
    assert "index" in rendered.split("Stages")[1]


def test_importing_the_cli_needs_no_conda_only_package():
    """Run in a subprocess: this test session has already imported most of them.

    Checks two separate things in the one subprocess, because spawning a second
    interpreter costs more than either assertion: nothing conda-only is reachable, and
    dask is not imported either. The latter is a startup-latency contract, not a
    packaging one — see blockrun's test_lazy_dask.
    """
    probe = CONDA_ONLY + ["dask", "distributed"]
    code = (
        "import sys; import neu_morpho.cli; "
        f"print(','.join(m for m in {probe!r} "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    got = set(filter(None, out.split(",")))

    conda = sorted(got & set(CONDA_ONLY))
    assert not conda, (
        f"importing neu_morpho.cli now pulls in {conda}. The docs build installs from "
        f"PyPI only, and these are conda-only on flyem-forge, so this would break it. "
        f"Defer the import into the function that needs it.")
    heavy = sorted(got & {"dask", "distributed"})
    assert not heavy, (
        f"importing neu_morpho.cli now pulls in {heavy}, which is ~1 s added to every "
        f"invocation — including `neu-morpho progress`, which only reads JSONL. Import "
        f"start_dask inside the branch that starts a cluster, not at module scope.")
