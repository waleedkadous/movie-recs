files_dir = "s3://waleed-movies"
#files_dir = "/home/ec2-user/images"

import os
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image, ImageColor, ImageDraw, ImageFont
import json

import ray
from ray.data.datasource import ImageFolderDatasource
from ray.air.util.tensor_extensions.pandas import TensorArray, TensorArrayElement
from ray.train.torch import TorchCheckpoint, TorchPredictor
from ray.train.batch_predictor import BatchPredictor


import torch
from torchvision import transforms
from torchvision.models.detection.ssd import ssd300_vgg16

from util import draw_bounding_boxes, save_images

def read_data_from_s3() -> ray.data.Dataset:
    def convert_to_pandas(byte_item_list):
        preprocess = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

        images = [Image.open(BytesIO(byte_item)).convert('RGB') for byte_item in byte_item_list]
        images = [preprocess(image) for image in images]
        images = [np.asarray(image) for image in images]

        return pd.DataFrame({"image": TensorArray(images)})

    # What is parallelism magic number?
    # TODO: Support reading images of different sizes in ImageFolderDatasource
    dataset = ray.data.read_binary_files(paths=files_dir)

    # TODO: Debug object spilling behavior.
    #dataset = dataset.limit(int(0.25*dataset.count()))
    dataset = dataset.map_batches(convert_to_pandas)
    
    return dataset


def batch_predict(dataset: ray.data.Dataset) -> ray.data.Dataset:
    model = ssd300_vgg16(pretrained=True)


    # # TODO: TorchCheckpoint should accept a model or a model state dict
    ckpt = TorchCheckpoint.from_model(model=model)

    class MyPredictor(TorchPredictor):
        def _predict_pandas(self, df, dtype):
            images = df["image"].to_numpy()
            torch_images = super()._arrays_to_tensors(images, dtype)
            output = super()._model_predict(torch_images)
            for d in output:
                for k, v in d.items():
                    d[k] = v.cpu().detach().numpy()
            df = pd.DataFrame(output)
            return df


    predictor = BatchPredictor.from_checkpoint(ckpt, MyPredictor)
    results = predictor.predict(dataset, num_gpus_per_worker=1, batch_size=96, keep_columns=["image"])
    return results
    
    
def visualize_objects(prediction_outputs: ray.data.Dataset):
    score_threshold = 0.8

    save_dir = "./object_detections"

    
    # Let's visualize the first 100 movie posters.
    images_to_draw = prediction_outputs.limit(100).map(draw_bounding_boxes)

    save_images(images_to_draw.iter_rows(), save_dir=save_dir)
    
        
        
if __name__ == "__main__":
    #ray.init("anyscale://workspace-project-sagemaker-demo/workspace-cluster-sagemaker-demo")
    dataset = read_data_from_s3()
    prediction_results = batch_predict(dataset)
    visualize_objects(prediction_results)

