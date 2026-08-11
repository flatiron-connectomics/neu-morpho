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
    from em_seg_morpho.cli import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "em-morpho"


def test_every_subcommand_is_reachable_from_the_parser():
    from em_seg_morpho import cli

    parser = cli.build_parser()
    subs = next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices
    assert set(subs) == {"run", "progress", "run-report"}
    for name, sub in subs.items():
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_run_validation_still_reports_against_the_run_subparser():
    """`build_parser` hands `_parse_args` the run subparser so its errors keep naming
    `em-morpho run ...` rather than the top-level usage."""
    from em_seg_morpho import cli

    parser = cli.build_parser()
    assert getattr(parser, "_run_parser", None) is not None
    assert parser._run_parser.prog.endswith("run")


def test_importing_the_cli_needs_no_conda_only_package():
    """Run in a subprocess: this test session has already imported most of them."""
    code = (
        "import sys; import em_seg_morpho.cli; "
        f"print(','.join(m for m in {CONDA_ONLY!r} "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", (
        f"importing em_seg_morpho.cli now pulls in {out}. The docs build installs from "
        f"PyPI only, and these are conda-only on flyem-forge, so this would break it. "
        f"Defer the import into the function that needs it.")
