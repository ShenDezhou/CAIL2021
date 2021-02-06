# coding=utf-8
"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint.
Support Tensorflow version 2.4.1.
"""

import os
import argparse
import torch
import numpy as np
import tensorflow as tf
from transformers import BertForMaskedLM, TFBertForMaskedLM



def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="model name e.g. xlnet-tiny-chinese")
    parser.add_argument("--cache_dir",
                        type=str,
                        default=None,
                        required=False,
                        help="Directory containing pytorch model")
    parser.add_argument("--pytorch_model_path",
                        type=str,
                        required=True,
                        help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir",
                        type=str,
                        required=True,
                        help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)


    # Load the PyTorch model in TensorFlow
    tf_model = TFBertForMaskedLM.from_pretrained(args.cache_dir, from_pt=True)

    # Save the TensorFlow model
    tf_model.save_pretrained(args.tf_cache_dir)


if __name__ == "__main__":
    main()