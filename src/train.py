"""
Fine-tune BERT for customer complaint PRODUCT classification.

Dataset:
    data/complaints.xlsx

Text column:
    Consumer complaint narrative

Target column:
    Product

Run:
    .\.venv\Scripts\python.exe training\train.py

Fast presentation/demo:
    .\.venv\Scripts\python.exe training\train.py --epochs 1 --max_length 128
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "complaints.xlsx"

OUT = BASE / "model" / "complaint_bert"


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )

    # ORIGINAL BERT
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased"
    )

    args = parser.parse_args()


    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_excel(DATA)

    print("Original records:", len(df))


    # ========================================================
    # 2. SELECT REQUIRED COLUMNS
    # ========================================================

    df = df[
        [
            "Consumer complaint narrative",
            "Product"
        ]
    ].dropna()


    # ========================================================
    # 3. CLEAN TEXT
    # ========================================================

    df["text"] = (
        df["Consumer complaint narrative"]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


    # Remove extremely short complaints

    df = df[
        df["text"].str.len() >= 10
    ].reset_index(drop=True)


    print(
        "Usable complaint records:",
        len(df)
    )


    # ========================================================
    # 4. ENCODE PRODUCT LABELS
    # ========================================================

    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(
        df["Product"]
    )


    # Create label mapping

    labels = {
        str(i): str(name)
        for i, name
        in enumerate(label_encoder.classes_)
    }


    print(
        "Number of product categories:",
        len(labels)
    )


    # ========================================================
    # 5. TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    print("\nSplitting dataset...")


    # 80% Training
    # 20% Temporary

    train, temp = train_test_split(
        df[
            [
                "text",
                "label"
            ]
        ],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )


    # 10% Validation
    # 10% Test

    # IMPORTANT:
    # Do not use stratify here because one of the
    # classes has too few examples.

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42
    )


    print(
        "Training samples:",
        len(train)
    )

    print(
        "Validation samples:",
        len(val)
    )

    print(
        "Test samples:",
        len(test)
    )


    # ========================================================
    # 6. LOAD BERT TOKENIZER
    # ========================================================

    print("\nLoading BERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )


    # ========================================================
    # 7. TOKENIZATION
    # ========================================================

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )


    print("\nTokenizing training data...")

    train_ds = (
        Dataset
        .from_pandas(
            train,
            preserve_index=False
        )
        .map(
            tokenize,
            batched=True
        )
    )


    print("Tokenizing validation data...")

    val_ds = (
        Dataset
        .from_pandas(
            val,
            preserve_index=False
        )
        .map(
            tokenize,
            batched=True
        )
    )


    print("Tokenizing test data...")

    test_ds = (
        Dataset
        .from_pandas(
            test,
            preserve_index=False
        )
        .map(
            tokenize,
            batched=True
        )
    )


    # ========================================================
    # 8. FORMAT DATA FOR PYTORCH
    # ========================================================

    columns = [
        "input_ids",
        "attention_mask",
        "label"
    ]


    train_ds.set_format(
        "torch",
        columns=columns
    )

    val_ds.set_format(
        "torch",
        columns=columns
    )

    test_ds.set_format(
        "torch",
        columns=columns
    )


    # ========================================================
    # 9. LOAD ORIGINAL BERT MODEL
    # ========================================================

    print("\nLoading BERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(

        args.model_name,

        num_labels=len(labels),

        id2label={
            int(k): v
            for k, v in labels.items()
        },

        label2id={
            v: int(k)
            for k, v in labels.items()
        }
    )


    # ========================================================
    # 10. EVALUATION METRICS
    # ========================================================

    def compute_metrics(eval_pred):

        logits, y_true = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1
        )


        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                predictions,
                average="weighted",
                zero_division=0
            )
        )


        accuracy = accuracy_score(
            y_true,
            predictions
        )


        return {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1
        }


    # ========================================================
    # 11. TRAINING CONFIGURATION
    # ========================================================

    training_args = TrainingArguments(

        output_dir=str(
            BASE /
            "results" /
            "trainer"
        ),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        learning_rate=2e-5,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_steps=50,

        report_to="none",

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        fp16=True
    )


    # ========================================================
    # 12. CREATE TRAINER
    # ========================================================

    # IMPORTANT:
    # tokenizer=tokenizer is intentionally NOT used here.
    # Newer Transformers versions reject that argument.

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        compute_metrics=compute_metrics
    )


    # ========================================================
    # 13. TRAIN MODEL
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STARTING BERT TRAINING")
    print("=" * 60)
    print("\n")

    trainer.train()


    # ========================================================
    # 14. TEST MODEL
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EVALUATING MODEL ON TEST DATA")
    print("=" * 60)
    print("\n")


    test_metrics = trainer.evaluate(
        test_ds
    )


    # ========================================================
    # 15. PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST METRICS")
    print("=" * 60)

    print(
        json.dumps(
            test_metrics,
            indent=4
        )
    )


    # ========================================================
    # 16. SAVE MODEL
    # ========================================================

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    trainer.save_model(
        str(OUT)
    )


    # ========================================================
    # 17. SAVE TOKENIZER
    # ========================================================

    tokenizer.save_pretrained(
        str(OUT)
    )


    # ========================================================
    # 18. SAVE LABEL MAPPING
    # ========================================================

    (
        OUT /
        "labels.json"
    ).write_text(

        json.dumps(
            labels,
            indent=4
        ),

        encoding="utf-8"
    )


    # ========================================================
    # 19. SAVE TEST METRICS
    # ========================================================

    results_folder = (
        BASE /
        "results"
    )

    results_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    (
        results_folder /
        "test_metrics.json"
    ).write_text(

        json.dumps(
            test_metrics,
            indent=4
        ),

        encoding="utf-8"
    )


    # ========================================================
    # 20. FINISHED
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "\nModel saved to:"
    )

    print(OUT)

    print(
        "\nMetrics saved to:"
    )

    print(
        results_folder /
        "test_metrics.json"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
