"""Operator tools exposed as subcommands: `em-morpho progress` / `run-report`.

They live inside the package so they ship with it — a tool that only exists in a
source checkout is one an installed command cannot offer. Underscored because they
are reached through the CLI, not imported as an API; each still has a ``main(argv)``
so it can be run directly during development.
"""
