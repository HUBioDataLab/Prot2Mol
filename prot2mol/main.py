"""Single meta entrypoint for Prot2Mol tasks.

Usage:
  python prot2mol/main.py train [args...]
  python prot2mol/main.py grpo [args...]
  python prot2mol/main.py generate [args...]
"""

import importlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

COMMANDS = {
    "train": "prot2mol.training.pretrain",
    "grpo": "prot2mol.training.grpo_train",
    "generate": "prot2mol.inference.produce_molecules",
}


def _print_help() -> None:
    print("Prot2Mol meta entrypoint")
    print("")
    print("Commands:")
    print("  train     Run training pipeline")
    print("  grpo      Run FusionDTI-rewarded GRPO post-training")
    print("  generate  Run molecule generation pipeline")
    print("")
    print("Examples:")
    print("  python prot2mol/main.py train --help")
    print("  python prot2mol/main.py train --config prot2mol/configs/train.yaml")
    print("  python prot2mol/main.py grpo --config prot2mol/configs/grpo.yaml")
    print("  python prot2mol/main.py generate --help")
    print("  python prot2mol/main.py generate --config prot2mol/configs/generate.yaml")


def _dispatch(command: str, args):
    module_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module '{module_name}' does not expose a main() function")

    original_argv = sys.argv
    try:
        sys.argv = [f"prot2mol {command}"] + list(args)
        return module.main()
    finally:
        sys.argv = original_argv


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print("")
        _print_help()
        return 2

    _dispatch(command, sys.argv[2:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
