"""Record which configuration produced an SPPO data round, so stale data is never reused."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from prompt_utils import RANKER, TEMPERATURE, TOP_P, load_prompt_dataset, num_shards, write_json_atomic

SPEC_NAME = "data_spec.json"
MODEL_TEXT_FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja")
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin")
DATA_SUFFIXES = (".json", ".jsonl", ".parquet", ".arrow", ".csv")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model):
    """Config/tokenizer hashes plus weight size and mtime; a hub id is kept as is."""
    if not os.path.isdir(model):
        return model
    root = Path(os.path.realpath(model))
    files = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name in MODEL_TEXT_FILES:
            files[path.name] = file_sha256(path)
        elif path.name.endswith(MODEL_WEIGHT_SUFFIXES):
            stat = path.stat()
            files[path.name] = [stat.st_size, stat.st_mtime_ns]
    return {"path": str(root), "files": files}


def data_identity(prompts):
    if os.path.isfile(prompts):
        return {"path": os.path.realpath(prompts), "sha256": file_sha256(prompts)}
    if os.path.isdir(prompts):
        root = Path(os.path.realpath(prompts))
        files = {
            str(p.relative_to(root)): file_sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in DATA_SUFFIXES
        }
        return {"path": str(root), "files": files}
    return prompts


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def build_spec(args, num_prompts):
    spec = {
        "model": model_identity(args.model),
        "prompts": data_identity(args.prompts),
        "num_prompts": num_prompts,
        "pairs": args.pairs,
        "maxlen": args.maxlen,
        "max_model_len": args.max_model_len,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "ranker": RANKER,
    }
    return json.loads(json.dumps(spec, sort_keys=True))


def prepare(args):
    gen_dir, rank_dir, score_dir = Path(args.gen_dir), Path(args.rank_dir), Path(args.score_dir)
    num_prompts = len(load_prompt_dataset(args.prompts))
    spec = build_spec(args, num_prompts)

    finished = read_json(score_dir / SPEC_NAME)
    if finished == {"spec": spec, "sim_model": args.sim_model} and (score_dir / "train.parquet").is_file():
        print("done")
        return

    record = read_json(gen_dir / SPEC_NAME)
    if record is None or record.get("spec") != spec:
        occupied = [d for d in (gen_dir, rank_dir, score_dir) if d.is_dir() and any(d.iterdir())]
        if occupied and not args.force:
            sys.exit(
                f"[dataset_spec] {', '.join(map(str, occupied))} hold data from a different or unrecorded "
                "configuration. Delete them, or set FORCE_REGENERATE=1 to delete them and regenerate."
            )
        for directory in occupied:
            print(f"[dataset_spec] deleting {directory}", file=sys.stderr)
            shutil.rmtree(directory)
        record = {"spec": spec, "frac_len": -(-num_prompts // args.num_gpus)}
        gen_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(gen_dir / SPEC_NAME, record)
    else:
        print(f"[dataset_spec] resuming {gen_dir} (frac_len={record['frac_len']})", file=sys.stderr)

    print(f"run {record['frac_len']} {num_shards(num_prompts, record['frac_len'])}")


def finalize(args):
    gen_dir, score_dir = Path(args.gen_dir), Path(args.score_dir)
    record = read_json(gen_dir / SPEC_NAME)
    if record is None:
        sys.exit(f"[dataset_spec] {gen_dir / SPEC_NAME} is missing; run prepare first.")
    parquet = score_dir / "train.parquet"
    if not parquet.is_file():
        sys.exit(f"[dataset_spec] {parquet} is missing.")
    if args.sim_model:
        import pyarrow.parquet as pq

        if "sim_score" not in pq.read_schema(parquet).names:
            sys.exit(f"[dataset_spec] {parquet} has no sim_score column.")
    write_json_atomic(score_dir / SPEC_NAME, {"spec": record["spec"], "sim_model": args.sim_model})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--pairs", type=int, required=True)
    p.add_argument("--maxlen", type=int, required=True)
    p.add_argument("--max_model_len", type=int, required=True)
    p.add_argument("--num_gpus", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=prepare)

    f = sub.add_parser("finalize")
    f.set_defaults(func=finalize)

    for sp in (p, f):
        sp.add_argument("--gen_dir", required=True)
        sp.add_argument("--score_dir", required=True)
        sp.add_argument("--sim_model", default=None)
    p.add_argument("--rank_dir", required=True)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
