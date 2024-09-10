from .api import *
from .utils import labeler
import argparse
import time
import logging
import subprocess
import json

def main():
    parser = argparse.ArgumentParser(description="Model management script")

    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    model_parser = subparsers_action.add_parser("model", help="Model-related actions")

    subparsers_model = model_parser.add_subparsers(dest="subaction", help="Model subactions")

    list_parser = subparsers_model.add_parser("list", help="List your models")

    summary_parser = subparsers_model.add_parser("summarize", help="Summarize a model")
    summary_parser.add_argument("--model_name", required=True, help="Name of the model to provide a summary")

    download_parser = subparsers_model.add_parser("download", help="Download and convert a model")
    download_parser.add_argument("--model_name", required=True, help="Name of the model to delete")
    download_parser.add_argument("--model_format", required=True, help="of the model to delete")

    delete_parser = subparsers_model.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("--model_name", required=True, help="Name of the model to delete")

    classify_parser = subparsers_action.add_parser("classify", help="Classifier-related actions")
    classify_parser.add_argument("--model_name", required=True, help="Name of the model")
    classify_parser.add_argument("--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'")
    classify_parser.add_argument("--model_size", required=True, help="Integer value for model size from 1=small up to 5=large")
    classify_parser.add_argument("--hf_dataset", required=False, default=None, help="(Optional) Name of the HuggingFace dataset to use for training")

    detect_parser = subparsers_action.add_parser("detect", help="Detector-related actions")
    detect_parser.add_argument("--model_name", required=True, help="Name of the model")
    detect_parser.add_argument("--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'")
    detect_parser.add_argument("--model_size", required=True, help="Integer value for model size from 1=small up to 5=large")
    detect_parser.add_argument("--hf_dataset", required=False, default=None, help="(Optional) Name of the HuggingFace dataset to use for training")

    generate_parser = subparsers_action.add_parser("generate", help="Generator-related actions")
    generate_parser.add_argument("--model_name", required=True, help="Name of the model")
    generate_parser.add_argument("--hf_dataset", required=True, default=None, help="Name of the HuggingFace dataset to use for training")

    deploy_parser = subparsers_action.add_parser("deploy", help="Deploy or tear down a model using Docker Compose")
    deploy_parser.add_argument("--model_name", required=True, help="Name of the model to deploy or tear down")
    deploy_parser.add_argument("command", nargs='?', default="up", choices=['up', 'down'], help="Specify 'up' to deploy or 'down' to tear down the model")

    infer_parser = subparsers_action.add_parser("infer", help="Submit an inference request to Triton server")
    infer_parser.add_argument("--model_name", type=str, required=True, help="Model name for the inference")
    infer_parser.add_argument("--prompt", type=str, required=True, help="Prompt to be used for the inference")
    infer_parser.add_argument("--server_url", type=str, default="localhost:8000", help="URL of the Triton Inference Server")
    infer_parser.add_argument("--model_version", type=str, default="1", help="Version of the model")

    user_parser = subparsers_action.add_parser("user", help="User-related actions")

    subparsers_user = user_parser.add_subparsers(dest="subaction", help="User subactions")
    profile_parser = subparsers_user.add_parser("profile", help="Get user profile")
    credits_parser = subparsers_user.add_parser("credits", help="Get user credits and subscription info if it exists")
    
    utils_parser = subparsers_action.add_parser("utils", help="Utility actions")
    
    subparsers_utils = utils_parser.add_subparsers(dest="subaction", help="Utility subactions")

    labeler_parser = subparsers_utils.add_parser("label", help="Extract labels from a directory of images")
    labeler_parser.add_argument("--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'")
    labeler_parser.add_argument("--image_dir", required=True, help="Directory of images in '.jpg', '.jpeg', '.png' format")
    labeler_parser.add_argument("--model_name", default=None, help="Name of the model")

    args = parser.parse_args()

    if args.action == "model":
        if args.subaction == "list":
            models = list_models()
            logging.info(json.dumps(models, indent=2))    
        elif args.subaction == "summarize":
            model_summary = get_model_summary(args.model_name)
            logging.info(json.dumps(model_summary, indent=2))
        elif args.subaction == "delete":
            deleted_model = delete_model(args.model_name)
            logging.info(json.dumps(deleted_model, indent=2))
        elif args.subaction == "download":
            downloaded_model = download_model(args.model_name, args.model_format)
            logging.info(json.dumps(downloaded_model, indent=2))
        else:
            logging.info("Invalid argument for 'model'")

    elif args.action == "classify":
        labels = args.labels.split(",")
        labels = [x.strip() for x in labels]
        training_classifier = train_classifier(args.model_name, labels, args.model_size, args.hf_dataset)
        logging.info(json.dumps(training_classifier, indent=2))

    elif args.action == "detect":                                                                            
        labels = args.labels.split(",")
        labels = [x.strip() for x in labels]
        training_detector = train_detector(args.model_name, labels, args.model_size, args.hf_dataset)
        logging.info(json.dumps(training_detector, indent=2))

    elif args.action == "generate":                                                                            
        training_generator = train_generator(args.model_name, args.hf_dataset)
        logging.info(json.dumps(training_generator, indent=2))

    elif args.action == "deploy":
        deploy_model(args.model_name, args.command)

    elif args.action == "infer":
        result, time_elapsed = run_inference(args.model_name, args.prompt, args.server_url, args.model_version)
        logging.info(f"Inference time: {time_elapsed:.4f} seconds")
        logging.info(json.dumps(result, indent=2))

    elif args.action == "user":
        if args.subaction == "profile":
            profile = get_user_profile()
            logging.info(profile)
        elif args.subaction == "credits":
            user_credits = get_user_credits()
            logging.info(json.dumps(user_credits, indent=2))
        else:
            logging.info("Invalid argument for 'user'")

    elif args.action == "utils":
        if args.subaction == "label":
            labels = args.labels.split(",")
            labels = [x.strip() for x in labels]
            results = labeler(labels, args.image_dir, args.model_name)
            logging.info(results)
        else:
            logging.info("Invalid argument for 'utils'")

    else:
        logging.info("Invalid action")

if __name__ == "__main__":
    main()
