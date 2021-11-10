import numpy as np
import os
import pandas as pd
import subprocess
import sys
import argparse
import logging


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--transformers_version", type=str)
    parser.add_argument("--pytorch_version", type=str)

    args, _ = parser.parse_known_args()

    install(f"torch=={args.pytorch_version}")
    install(f"transformers=={args.transformers_version}")
    install("datasets[s3]")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset(args.dataset_name, split=["train", "test"])
    test_dataset = test_dataset.shuffle().select(range(1000))  # smaller the size for test dataset to 1k

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset.save_to_disk("/opt/ml/processing/train")
    test_dataset.save_to_disk("/opt/ml/processing/test")
