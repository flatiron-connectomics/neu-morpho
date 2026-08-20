"""``python -m neu_morpho`` — the same entry point as the ``neu-morpho`` command.

Worth having as well as the console script: ``python -m`` works from a source
checkout without the package being installed, and makes it unambiguous which
interpreter is running when several environments are on PATH.
"""

from neu_morpho.cli import main

raise SystemExit(main())
