"""Register the SO-101 experiment with the framework's config system.

Configs do not register by being present on disk; an explicit import in
`configs/base/config.py` is what makes the experiment resolvable, and without it the
checkpoint fails with a hydra MissingConfigException that never mentions SO-101.

The imports live inside a function, so this inserts beside its siblings rather than appending
at end of file, and asserts the anchor exists so an upstream layout change fails the build
instead of producing an image that cannot load its own checkpoint.
"""

from pathlib import Path

TARGET = Path("/opt/cosmos-framework/cosmos_framework/configs/base/config.py")
ANCHOR = "configs.base.experiment.action.posttrain_config."
MODULE = "cosmos_framework.configs.base.experiment.action.posttrain_config.so101_v7_full_experiment"


def main() -> None:
    lines = TARGET.read_text().splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if ANCHOR in l and l.lstrip().startswith("import ")]
    assert hits, "no posttrain_config import block found; upstream layout changed"
    if any(MODULE in l for l in lines):
        print("already registered")
        return
    i = hits[-1]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i + 1, f"{indent}import {MODULE}  # noqa: F401\n")
    TARGET.write_text("".join(lines))
    print(f"registered so101_v7_full_experiment after line {i + 1}")


if __name__ == "__main__":
    main()
