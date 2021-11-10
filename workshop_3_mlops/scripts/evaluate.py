"""Evaluation script for measuring mean squared error."""

import subprocess
import sys
import json
import logging
import pathlib
import tarfile
import os

import numpy as np
import pandas as pd


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="./hf_model")

    logger.debug(os.listdir("./hf_model"))

    with open("./hf_model/evaluation.json") as f:
        eval_result = json.load(f)

    logger.debug(eval_result)
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(eval_result))
