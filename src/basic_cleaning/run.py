#!/usr/bin/env python
"""
Download data, perform data cleaning, and update data
"""
import argparse
import logging
import tempfile

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """Download data, perform data cleaning, and update data in wandb.

    Args:
        args: command line arguments

    Returns:
        None
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info("Cleaning data")
    # Drop outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df["last_review"] = pd.to_datetime(df["last_review"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Saving cleaned data")
        tmp_path = f"{tmp_dir}/clean_sample.csv"
        df.to_csv(tmp_path, index=False)

        artifact = wandb.Artifact(args.output_artifact, type=args.output_type, description=args.output_description)
        artifact.add_file(tmp_path)
        run.log_artifact(artifact)

        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Cleaning")

    parser.add_argument("--input_artifact", type=str, help="Fully-qualified name for the input artifact", required=True)

    parser.add_argument(
        "--output_artifact", type=str, help="Fully-qualified name for the output artifact", required=True
    )

    parser.add_argument("--output_type", type=str, help="Type of the output artifact", required=True)

    parser.add_argument("--output_description", type=str, help="Description of the output artifact", required=True)

    parser.add_argument("--min_price", type=float, help="Min price considered for data analysis", required=True)

    parser.add_argument("--max_price", type=float, help="Max price considered for data analysis", required=True)

    args = parser.parse_args()

    go(args)
