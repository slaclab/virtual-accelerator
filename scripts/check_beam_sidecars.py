"""Pre-commit hook: every ``beams/*.h5`` file must have a matching ``.h5.meta.json`` sidecar.

Invoked with a list of paths (staged ``.h5`` files under ``virtual_accelerator/beams/``)
and exits non-zero if any of them lack a sidecar.
"""

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    missing = []
    for arg in argv:
        h5 = Path(arg)
        sidecar = h5.with_name(h5.name + ".meta.json")
        if not sidecar.exists():
            missing.append(f"{h5}: missing sidecar {sidecar.name}")
    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
