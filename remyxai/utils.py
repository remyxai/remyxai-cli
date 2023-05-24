import os
import csv
import json
import time
import zipfile
from tqdm import tqdm

import onnx
import onnxruntime as ort

import numpy as np
from PIL import Image
from ast import literal_eval

from .api import *

# Utilities

def load_image(image_path, img_height=224, img_width=224):
    img = Image.open(image_path).convert('RGB')
    img = img.resize([img_height, img_width])

    img = np.asarray(img, dtype='float32') / 255
    # Return a scaled array between -1 and 1
    return img * 2 - 1

class RemyxModel(object):
    def __init__(self, model_assets_path):
        self.model_file = [x for x in os.listdir(model_assets_path) if x.endswith(".onnx")][0]
        self.model_path = os.path.join(model_assets_path, self.model_file)
        metadata_file = os.path.join(model_assets_path, "metadata.csv")

        # Parse model metadata
        with open(metadata_file, "r") as meta_file:
            reader = csv.reader(meta_file, delimiter=",")
            metadata = dict(zip(*reader))
            # formatting
            metadata["input_shape"] = literal_eval(metadata["input_shape"])

        self.model = ort.InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.img_height = metadata["input_shape"][1]
        self.img_width = metadata["input_shape"][2]
        self.labels = metadata["labels"].split("|")

    def __call__(self, inputs):
        """
        Run inference on a single image path or list of image paths.
        inputs: str or list of str
        """
        if isinstance(inputs, str):
            img = [load_image(inputs, self.img_height, self.img_width)]
        elif isinstance(inputs, list):
            img = [load_image(x, self.img_height, self.img_width) for x in inputs]
        else:
            return "Format not recognized. Please pass a string or list of strings."

        preds = self.model.run([self.output_name], {self.input_name: img})[0]
        pred_idx = np.argmax(preds, axis=1)
        pred_classes = np.take(self.labels, pred_idx, axis=0).tolist()
        results = [{"file": f, "label": l} for f,l in list(zip(inputs, pred_classes))]
        return results

    def predict(self, inputs):
        return self(inputs)

def process_images_in_directory(directory, processor, chunk_size=10):
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
    
    results = []
    if os.path.isfile('processed_results.json'):
        with open('processed_results.json', 'r') as json_file:
            results = json.load(json_file)

    for i in tqdm(range(0, len(image_files), chunk_size), desc='Processing images'):
        chunk = image_files[i:i+chunk_size]
        processed_chunk = processor(chunk)
        results.extend(processed_chunk)

        with open('processed_results.json', 'w') as json_file:
            json.dump(results, json_file)
            
        print(f"Processed {min(i+chunk_size, len(image_files))} out of {len(image_files)} images")
        
    return 'Results are stored in ./processed_results.json'


def labeler(labels: list, image_dir: str, model_name=None):
    """
    Wrapper to train a custom image classifier, 
    download model, and run inference on a folder of images.
    """
    # Step 1: Train a model if it doesn't exist
    if model_name is None:
        model_name = "labeler_{}".format("_".join(labels))
    status = get_model_summary(model_name)["message"][0]["status"]

    if status == "NOT_FOUND":
        print("Model is training, please wait...")
        train_classifier(model_name, labels, "3")

    while status != "FINISHED":
        status = get_model_summary(model_name)["message"][0]["status"]
        print(f'Current model status: {status}')
        if status == "RUNNING":
            # Check back in 5 minutes:
            time.sleep(300)
        elif status == "FAILED":
            print("Model training failed. Please try again.")
            return

    print("Model is ready for inference.")
    print("Downloading model...")
    download_message = download_model(model_name=model_name, model_format="onnx")

    print(download_message)
    print("Preparing for inference...")                                                                                                                                                                     
    # Step 2: Decompress model assets, load onnx model  
    model_assets_path = os.path.join(os.getcwd(), model_name)
    model_assets_zip = model_assets_path + ".zip"
    with zipfile.ZipFile(model_assets_zip,"r") as zip_ref:
        zip_ref.extractall(model_assets_path)
    model = RemyxModel(model_assets_path)

    # Step 3: Process images
    process_message = process_images_in_directory(image_dir, model, chunk_size=8)
    return process_message
