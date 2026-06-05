from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA train -> update .env -> run eval in one command"
    )
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--golden-path", default="data/golden_set.json")
    parser.add_argument("--corpus-path", default="data/synthetic_corpus.jsonl")
    parser.add_argument("--output-dir", default="models/reranker_lora")
    parser.add_argument("--merged-output-dir", default="models/reranker_lora_merged")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local model/cache during LoRA training",
    )
    parser.add_argument(
        "--env-path",
        default=".env",
        help="Path to environment file to update",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip LoRA training step and only update .env + run eval",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after .env update",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def upsert_env_values(env_path: Path, values: dict[str, str]) -> None:
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    consumed: set[str] = set()
    updated_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            updated_lines.append(line)
            continue

        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in values:
            updated_lines.append(f"{key}={values[key]}")
            consumed.add(key)
        else:
            updated_lines.append(line)

    for key, value in values.items():
        if key not in consumed:
            updated_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "-m",
            "scripts.train_lora_reranker",
            "--base-model",
            args.base_model,
            "--golden-path",
            args.golden_path,
            "--corpus-path",
            args.corpus_path,
            "--output-dir",
            args.output_dir,
            "--merged-output-dir",
            args.merged_output_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--negatives-per-positive",
            str(args.negatives_per_positive),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--seed",
            str(args.seed),
            "--merge-adapter",
        ]
        if args.local_files_only:
            train_cmd.append("--local-files-only")

        print("[1/3] Training LoRA reranker...")
        run_cmd(train_cmd)
    else:
        print("[1/3] Training skipped by --skip-train")

    env_path = Path(args.env_path)
    runtime_values = {
        "RERANKER_MODEL_NAME": args.merged_output_dir,
        "RERANKER_LOCAL_FILES_ONLY": "true",
    }

    print(f"[2/3] Updating {env_path} with tuned reranker settings...")
    upsert_env_values(env_path, runtime_values)

    if not args.skip_eval:
        print("[3/3] Running evaluation...")
        run_cmd([sys.executable, "-m", "scripts.run_eval"])
    else:
        print("[3/3] Evaluation skipped by --skip-eval")

    print("\nDone. Runtime now points to tuned reranker model.")


if __name__ == "__main__":
    main()
