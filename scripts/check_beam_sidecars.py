"""Pre-commit hook: every ``beams/*.h5`` file must have a matching ``.meta.json`` sidecar.

Invoked with a list of paths (staged ``.h5`` files under ``virtual_accelerator/beams/``)
and exits non-zero if any of them lack a sidecar in the Git index (tracked or staged).
"""

import subprocess
import sys
from pathlib import Path


def index_has(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--error-unmatch", "--", str(path)],
        capture_output=True,
    )
    return result.returncode == 0


def main(argv: list[str]) -> int:
    missing = []
    for arg in argv:
        h5 = Path(arg)
        sidecar = h5.with_suffix(".meta.json")
        if not index_has(sidecar):
            missing.append(f"{h5}: missing sidecar {sidecar.name} in the commit")
    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
