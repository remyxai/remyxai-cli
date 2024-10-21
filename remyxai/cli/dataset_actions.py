from remyxai.api.datasets import list_datasets, delete_dataset, download_dataset


def handle_dataset_action(args):
    """Handle dataset actions (list, delete)."""
    if args["action"] == "list":
        datasets = list_datasets()
        print(datasets)
    elif args["action"] == "delete":
        delete_dataset(args["dataset_name"])
    elif args["action"] == "download":
        download_dataset(args["dataset_name"])
    
