"""Local-only CLI. Exit 2 on invalid inputs; never label data scientifically verified."""
import argparse
import hashlib
from pathlib import Path
import sys
import yaml
from researchbench import __version__
from researchbench.adapters.files import ingest_config
from researchbench.core.schema import read_records, write_records
from researchbench.export.bundle import export_bundle


def parser():
    p = argparse.ArgumentParser(prog="researchbench", description="Reproducible ML/RL experiment analysis")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Map CSV/JSON/NPZ files into validated records")
    ing.add_argument("config", type=Path)
    ing.add_argument("--output", type=Path, required=True)
    val = sub.add_parser("validate", help="Validate records; optionally check current source hashes")
    val.add_argument("records", type=Path)
    val.add_argument("--check-sources", action="store_true")
    for name, arg in (("run", "config"), ("summarize", "records")):
        cmd = sub.add_parser(name, help="Export validated CSV, LaTeX, plots and provenance")
        cmd.add_argument(arg, type=Path)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--baseline", help="Model name for strict paired-seed ablations")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "ingest":
            records = ingest_config(args.config)
            if args.output.exists():
                raise ValueError("Output already exists; choose a new file")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_records(records, args.output)
        elif args.command == "validate":
            records = read_records(args.records)
            if args.check_sources:
                for r in records:
                    for source in r.sources:
                        actual = hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
                        if actual != source["sha256"]:
                            raise ValueError(f"Source hash mismatch: {source['path']}")
        else:
            records = ingest_config(args.config) if args.command == "run" else read_records(args.records)
            export_bundle(records, args.output, baseline=args.baseline)
        print(f"OK: {len(records)} runs; record contracts validated; scientific status: Unverified")
        return 0
    except (ValueError, KeyError, TypeError, OSError, IndexError, yaml.YAMLError) as exc:
        print(f"researchbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
