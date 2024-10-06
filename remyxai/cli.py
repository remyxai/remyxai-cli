import argparse
import time
import logging
import subprocess
import json
import sys
from actions import *


def main():
    parser = argparse.ArgumentParser(description="Model management script")

    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    model_parser = subparsers_action.add_parser("model", help="Model-related actions")

    subparsers_model = model_parser.add_subparsers(
        dest="subaction", help="Model subactions"
    )

    list_parser = subparsers_model.add_parser("list", help="List your models")

    summary_parser = subparsers_model.add_parser("summarize", help="Summarize a model")
    summary_parser.add_argument(
        "--model_name", required=True, help="Name of the model to provide a summary"
    )

    download_parser = subparsers_model.add_parser(
        "download", help="Download and convert a model"
    )
    download_parser.add_argument(
        "--model_name", required=True, help="Name of the model to delete"
    )
    download_parser.add_argument(
        "--model_format", required=True, help="of the model to delete"
    )

    delete_parser = subparsers_model.add_parser("delete", help="Delete a model")
    delete_parser.add_argument(
        "--model_name", required=True, help="Name of the model to delete"
    )

    classify_parser = subparsers_action.add_parser(
        "classify", help="Classifier-related actions"
    )
    classify_parser.add_argument(
        "--model_name", required=True, help="Name of the model"
    )
    classify_parser.add_argument(
        "--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'"
    )
    classify_parser.add_argument(
        "--model_size",
        required=True,
        help="Integer value for model size from 1=small up to 5=large",
    )
    classify_parser.add_argument(
        "--hf_dataset",
        required=False,
        default=None,
        help="(Optional) Name of the HuggingFace dataset to use for training",
    )

    detect_parser = subparsers_action.add_parser(
        "detect", help="Detector-related actions"
    )
    detect_parser.add_argument("--model_name", required=True, help="Name of the model")
    detect_parser.add_argument(
        "--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'"
    )
    detect_parser.add_argument(
        "--model_size",
        required=True,
        help="Integer value for model size from 1=small up to 5=large",
    )
    detect_parser.add_argument(
        "--hf_dataset",
        required=False,
        default=None,
        help="(Optional) Name of the HuggingFace dataset to use for training",
    )

    generate_parser = subparsers_action.add_parser(
        "generate", help="Generator-related actions"
    )
    generate_parser.add_argument(
        "--model_name", required=True, help="Name of the model"
    )
    generate_parser.add_argument(
        "--hf_dataset",
        required=True,
        default=None,
        help="Name of the HuggingFace dataset to use for training",
    )

    deploy_parser = subparsers_action.add_parser(
        "deploy", help="Deploy or tear down a model using Docker Compose"
    )
    deploy_parser.add_argument(
        "--model_name", required=True, help="Name of the model to deploy or tear down"
    )
    deploy_parser.add_argument(
        "command",
        nargs="?",
        default="up",
        choices=["up", "down"],
        help="Specify 'up' to deploy or 'down' to tear down the model",
    )

    infer_parser = subparsers_action.add_parser(
        "infer", help="Submit an inference request to Triton server"
    )
    infer_parser.add_argument(
        "--model_name", type=str, required=True, help="Model name for the inference"
    )
    infer_parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt to be used for the inference"
    )
    infer_parser.add_argument(
        "--server_url",
        type=str,
        default="localhost:8000",
        help="URL of the Triton Inference Server",
    )
    infer_parser.add_argument(
        "--model_version", type=str, default="1", help="Version of the model"
    )

    user_parser = subparsers_action.add_parser("user", help="User-related actions")

    subparsers_user = user_parser.add_subparsers(
        dest="subaction", help="User subactions"
    )
    profile_parser = subparsers_user.add_parser("profile", help="Get user profile")
    credits_parser = subparsers_user.add_parser(
        "credits", help="Get user credits and subscription info if it exists"
    )

    utils_parser = subparsers_action.add_parser("utils", help="Utility actions")

    subparsers_utils = utils_parser.add_subparsers(
        dest="subaction", help="Utility subactions"
    )

    labeler_parser = subparsers_utils.add_parser(
        "label", help="Extract labels from a directory of images"
    )
    labeler_parser.add_argument(
        "--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'"
    )
    labeler_parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory of images in '.jpg', '.jpeg', '.png' format",
    )
    labeler_parser.add_argument("--model_name", default=None, help="Name of the model")

    args = parser.parse_args()

    try:
        if args.action == "model":
            handle_model_action(args)
        elif args.action in ["classify", "detect", "generate"]:
            handle_training_action(args)
        elif args.action == "deploy":
            deploy_model(args.model_name, args.command)
        elif args.action == "infer":
            handle_inference(args)
        elif args.action == "user":
            handle_user_action(args)
        elif args.action == "utils":
            handle_utils_action(args)
        else:
            raise ValueError(f"Invalid action: {args.action}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
