from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainExample:
    question: str
    passage: str
    label: float


def load_corpus_sections(corpus_path: Path) -> dict[str, str]:
    section_map: dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            for section in doc.get("sections", []):
                key = f"{doc_id}#{section['section_id']}"
                section_map[key] = section.get("text", "")
    return section_map


def build_pairwise_dataset(
    golden_path: Path,
    corpus_path: Path,
    negatives_per_positive: int,
    seed: int,
) -> list[TrainExample]:
    rnd = random.Random(seed)
    section_map = load_corpus_sections(corpus_path)
    all_sections = list(section_map.items())

    golden_rows = json.loads(golden_path.read_text(encoding="utf-8"))
    examples: list[TrainExample] = []

    for row in golden_rows:
        question = row["question"]
        expected = row.get("expected_citations", [])
        positive_keys = [key for key in expected if key in section_map]
        if not positive_keys:
            continue

        positive_set = set(positive_keys)
        negative_pool = [item for item in all_sections if item[0] not in positive_set]

        for p_key in positive_keys:
            examples.append(
                TrainExample(question=question, passage=section_map[p_key], label=1.0)
            )

            if not negative_pool:
                continue

            sampled_negs = rnd.sample(
                negative_pool,
                k=min(negatives_per_positive, len(negative_pool)),
            )
            for _, neg_text in sampled_negs:
                examples.append(
                    TrainExample(question=question, passage=neg_text, label=0.0)
                )

    rnd.shuffle(examples)
    return examples


def to_hf_dataset(examples: list[TrainExample]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": ex.question,
                "passage": ex.passage,
                "label": ex.label,
            }
            for ex in examples
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune reranker for compliance RAG")
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base cross-encoder model",
    )
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
        help="Only load model/tokenizer from local cache or local path",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge LoRA weights into base model for direct CrossEncoder loading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = build_pairwise_dataset(
        golden_path=Path(args.golden_path),
        corpus_path=Path(args.corpus_path),
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    if len(examples) < 8:
        raise RuntimeError("Not enough training examples built from golden set.")

    split_idx = max(1, int(0.8 * len(examples)))
    train_ds = to_hf_dataset(examples[:split_idx])
    val_ds = to_hf_dataset(examples[split_idx:])

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            local_files_only=args.local_files_only,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=1,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load base model/tokenizer. "
            "If you are in a corporate SSL/proxy environment, pass a local model path to --base-model "
            "and add --local-files-only, or configure REQUESTS_CA_BUNDLE / HF trust store."
        ) from exc

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list[float]]:
        tok = tokenizer(
            batch["question"],
            batch["passage"],
            truncation=True,
            max_length=args.max_length,
        )
        tok["labels"] = batch["label"]
        return tok

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.merge_adapter:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        peft_loaded = PeftModel.from_pretrained(base_model, str(output_dir))
        merged_model = peft_loaded.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged model saved at: {merged_dir}")

    print(f"LoRA adapter saved at: {output_dir}")
    print("Set RERANKER_MODEL_NAME to merged model path for runtime inference.")


if __name__ == "__main__":
    main()
