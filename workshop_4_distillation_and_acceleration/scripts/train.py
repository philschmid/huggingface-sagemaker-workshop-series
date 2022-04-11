import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--dataset_id", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--hub_token", type=str, default=None)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # load datasets
    dataset = load_dataset(args.dataset_id, args.dataset_config)

    # process dataset
    def process(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=512)
        return tokenized_inputs

    tokenized_datasets = dataset.map(process, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("intent", "labels")

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    # create label2id, id2label dicts for nice outputs for the model
    labels = tokenized_datasets["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
        
    # define training args
    output_dir = Path("/opt/ml/output/data")
    training_args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        learning_rate=float(args.learning_rate),
        # logging & evaluation strategies
        logging_dir=f"{output_dir.as_posix()}/logs",
        logging_strategy="epoch",  # to get more information to TB
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=args.push_to_hub,
        hub_strategy="every_save",
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_dir.as_posix(), "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")

    # save best model, metrics and create model card
    if args.push_to_hub:
        trainer.create_model_card(model_name=args.hub_model_id)
        trainer.push_to_hub()
        
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
