from .api import *
import json
import logging
from .utils import labeler

def handle_model_action(args):
    if args.subaction == "list":
        models = list_models()
    elif args.subaction == "summarize":
        models = get_model_summary(args.model_name)
    elif args.subaction == "delete":
        models = delete_model(args.model_name)
    elif args.subaction == "download":
        models = download_model(args.model_name, args.model_format)
    else:
        raise ValueError(f"Invalid subaction for 'model': {args.subaction}")
    logging.info(json.dumps(models, indent=2))

def handle_training_action(args):
    labels = [x.strip() for x in args.labels.split(",")] if args.action != "generate" else None
    if args.action == "classify":
        result = train_classifier(args.model_name, labels, args.model_size, args.hf_dataset)
    elif args.action == "detect":
        result = train_detector(args.model_name, labels, args.model_size, args.hf_dataset)
    elif args.action == "generate":
        result = train_generator(args.model_name, args.hf_dataset)
    logging.info(json.dumps(result, indent=2))

def handle_inference(args):
    result, time_elapsed = run_inference(args.model_name, args.prompt, args.server_url, args.model_version)
    logging.info(f"Inference time: {time_elapsed:.4f} seconds")
    logging.info(json.dumps(result, indent=2))

def handle_user_action(args):
    if args.subaction == "profile":
        result = get_user_profile()
    elif args.subaction == "credits":
        result = get_user_credits()
    else:
        raise ValueError(f"Invalid subaction for 'user': {args.subaction}")
    logging.info(json.dumps(result, indent=2))

def handle_utils_action(args):
    if args.subaction == "label":
        labels = [x.strip() for x in args.labels.split(",")]
        results = labeler(labels, args.image_dir, args.model_name)
        logging.info(results)
    else:
        logging.info(f"Invalid subaction for 'utils': {args.subaction}")

def handle_myxmatch_action(args):
    if args.action == "list":
        result = list_myxboards()
    elif args.action == "summarize":
        if not args.match_name:
            raise ValueError("--match_name is required for summarize action")
        result = get_myxboard_summary(args.match_name)
    elif args.action == "delete":
        if not args.myxboard_name:
            raise ValueError("--myxboard_name is required for delete action")
        result = delete_myxboard(args.myxboard_name)
    else:
        raise ValueError(f"Invalid action for 'myxmatch': {args.action}")
    logging.info(json.dumps(result, indent=2))