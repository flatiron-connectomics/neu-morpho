"""``python -m em_seg_morpho`` — the same entry point as the ``em-seg-morpho`` command.

Worth having as well as the console script: ``python -m`` works from a source
checkout without the package being installed, and makes it unambiguous which
interpreter is running when several environments are on PATH.
"""

from em_seg_morpho.cli import main

raise SystemExit(main())
