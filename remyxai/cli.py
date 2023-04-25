from .api import *
import argparse
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="Model management script")

    # Define top-level actions
    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    # Define 'model' action
    model_parser = subparsers_action.add_parser("model", help="Model-related actions")

    # Define subactions for 'model' action
    subparsers_model = model_parser.add_subparsers(dest="subaction", help="Model subactions")

    # Define 'list' subaction
    list_parser = subparsers_model.add_parser("list", help="Create a new model")

    # Define 'download' subaction
    download_parser = subparsers_model.add_parser("download", help="Download and convert a model")
    download_parser.add_argument("--model_name", required=True, help="Name of the model to delete")
    download_parser.add_argument("--model_format", required=True, help="of the model to delete")

    # Define 'delete' subaction
    delete_parser = subparsers_model.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("--model_name", required=True, help="Name of the model to delete")

    # Define 'classify' subaction
    classify_parser = subparsers_action.add_parser("classify", help="Classifier-related actions")
    classify_parser.add_argument("--model_name", required=True, help="Name of the model")
    classify_parser.add_argument("--labels", required=True, help="Comma separated list of labels, e.g. 'cat,dog'")
    classify_parser.add_argument("--model_size", required=True, help="Integer value for model size from 1=small up to 5=large")

    # Define 'user' action
    user_parser = subparsers_action.add_parser("user", help="User-related actions")

    # Define subactions for 'user' action
    subparsers_user = user_parser.add_subparsers(dest="subaction", help="User subactions")
    profile_parser = subparsers_user.add_parser("profile", help="Get user profile")
    credits_parser = subparsers_user.add_parser("credits", help="Get user credits and subscription info if it exists")

    args = parser.parse_args()

    if args.action == "model":
        if args.subaction == "list":
            models = list_models()
            pprint(models)
        elif args.subaction == "delete":
            deleted_model = delete_model(args.model_name)
            pprint(deleted_model)
        elif args.subaction == "download":
            downloaded_model = download_model(args.model_name, args.model_format)
            print(downloaded_model)
        else:
            print("Invalid argument for 'model'")
    elif args.action == "classify":
        labels = args.labels.split(",")
        training_classifier = train_classifier(args.model_name, labels, args.model_size)
        pprint(training_classifier) 
    elif args.action == "user":
        if args.subaction == "profile":
            profile = get_user_profile()
            pprint(profile)
        elif args.subaction == "credits":
            user_credits = get_user_credits()
            pprint(user_credits)
        else:
            print("Invalid argument for 'user'")
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
