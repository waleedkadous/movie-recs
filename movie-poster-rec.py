files_dir = "s3://waleed-movies"
#files_dir = "/home/ec2-user/images"

import os
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import json

import ray
from ray.air import Checkpoint
from ray.data.context import DatasetContext
from ray.train.predictor import Predictor
from ray.train.torch import TorchPredictor
from ray.train.batch_predictor import BatchPredictor


import torch
from torchvision import transforms
from torchvision.models.detection.ssd import ssd300_vgg16

from util import draw_bounding_boxes, save_images


ctx = DatasetContext.get_current(); ctx.enable_tensor_extension_casting = False

def read_data_from_s3() -> ray.data.Dataset:
    def convert_to_pandas(byte_item_list):
        preprocess = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

        images = [Image.open(BytesIO(byte_item)).convert('RGB') for byte_item in byte_item_list]
        images = [preprocess(image) for image in images]
        images = [np.asarray(image) for image in images]

        return pd.DataFrame({"image": images})

    # TODO: Switch to ImageFolderDatasource, remove convert_to_pandas, and move torchvision transforms to a Preprocessor after https://anyscaleteam.slack.com/archives/C030DEV6QLU/p1659083614898059 is addressed
    # In particular if kwargs can be passed to the underlying `imageio.imread()` in `ImageFolderDatasource`, then we can specify `mode="RGB"` to match the `.convert("RGB")`
    # that we have in our convert_to_pandas_function. Otherwise, because we have a mix of black and white images and color images, the dimensions do not match causing torchvision transforms to complain.
    dataset = ray.data.read_binary_files(paths=files_dir)

    # TODO: Debug object spilling behavior.
    #dataset = dataset.limit(int(0.25*dataset.count()))
    dataset = dataset.map_batches(convert_to_pandas)
    
    return dataset


def batch_predict() -> ray.data.Dataset:
    dataset = read_data_from_s3()
    model = ssd300_vgg16(pretrained=True)

    ckpt = Checkpoint.from_dict({"model": model})
    
    # For a use case like this (model returns a non-standard output type like List[Dict[str, torch.Tensor]]) we would recommend users to implement their own Predictors. For demo purposes we can hide this Predictor implementation, but I don't think there is a good way we can generically support all output types in TorchPredictor directly.
    # See here for more information about the model inputs and outputs: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html
    class SSDPredictor(TorchPredictor):
        def _predict_pandas(self, df: pd.DataFrame, **kwargs):
            images = [torch.as_tensor(image).to("cuda") for image in df["image"].to_list()]
            self.model.eval()
            model_output = self.model(images)
            model_output = [{k: v.detach().cpu().numpy() for k, v in objects.items()} for objects in model_output]
            return pd.DataFrame(model_output)


    predictor = BatchPredictor.from_checkpoint(ckpt, SSDPredictor)
    results = predictor.predict(dataset, num_gpus_per_worker=1, batch_size=128, keep_columns=["image"])
    return results
    
    
def visualize_objects(prediction_outputs: ray.data.Dataset):
    score_threshold = 0.8

    save_dir = "./object_detections"

    
    # Let's visualize the first 100 movie posters.
    images_to_draw = prediction_outputs.limit(100).map(draw_bounding_boxes)

    save_images(images_to_draw.iter_rows(), save_dir=save_dir)
    
        
        
if __name__ == "__main__":
    prediction_results = batch_predict()
    visualize_objects(prediction_results)

