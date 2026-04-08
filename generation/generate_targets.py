from __future__ import annotations

import argparse

from generation.target_corpus import generateTargetCorpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline procedural target corpus")
    parser.add_argument("--config", required=True, help="Path to target corpus YAML config")
    parser.add_argument(
        "--workers",
        default=None,
        help="Worker count override. Use an integer or 'auto'.",
    )
    args = parser.parse_args()

    worker_override: int | str | None
    if args.workers is None:
        worker_override = None
    else:
        worker_override = args.workers if args.workers == "auto" else int(args.workers)

    manifest = generateTargetCorpus(
        args.config,
        workers=worker_override,
        showProgress=True,
    )
    print(f"Generated {manifest['count']} targets")
    print(f"Workers: {manifest['workers']}")
    print(f"Power mode: {manifest['powerMode']}")
    print(f"Curriculum counts: {manifest['curriculum']['counts']}")


if __name__ == "__main__":
    main()
