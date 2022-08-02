files_dir = "s3://waleed-movies"
#files_dir = "/home/ec2-user/images"

import os
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

import ray
from ray.data.context import DatasetContext
from ray.train.torch import TorchPredictor, TorchCheckpoint
from ray.train.batch_predictor import BatchPredictor


import torch
from torchvision import transforms
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import ssd300_vgg16

from util import VGG16Predictor, draw_bounding_boxes, save_images


ctx = DatasetContext.get_current()
ctx.enable_tensor_extension_casting = False


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

    # TODO: Switch to ImageFolderDatasource, remove convert_to_pandas,
    # and move torchvision transforms to a Preprocessor after
    # https://anyscaleteam.slack.com/archives/C030DEV6QLU/p1659083614898059 is addressed
    # In particular if kwargs can be passed to the underlying `imageio.imread()`
    # in `ImageFolderDatasource`, then we can specify `mode="RGB"` to match the `.convert("RGB")`
    # that we have in our convert_to_pandas_function.
    # Otherwise, because we have a mix of black and white images and color images,
    # the dimensions do not match causing torchvision transforms to complain.
    dataset = ray.data.read_binary_files(paths=files_dir)

    # TODO: Debug object spilling behavior.
    # For now, take 25% of all the images. Running batch prediction over the entire dataset
    # causes out-of-disk error on a ml.g4dn.12xlarge with 3GB of free disk space.
    dataset = dataset.limit(int(0.25 * dataset.count()))
    dataset = dataset.map_batches(convert_to_pandas)
    
    return dataset


def batch_predict() -> ray.data.Dataset:
    dataset = read_data_from_s3()

    ckpt = TorchCheckpoint.from_model(
        model=ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    )
    predictor = BatchPredictor.from_checkpoint(ckpt, VGG16Predictor)
    results = predictor.predict(
        dataset, num_gpus_per_worker=1, batch_size=32, keep_columns=["image"]
    )

    return results
    
    
def visualize_objects(prediction_outputs: ray.data.Dataset):
    # Let's visualize the first 100 movie posters.
    images_to_draw = prediction_outputs.limit(100).map(draw_bounding_boxes)
    save_dir = "./object_detections"
    save_images(images_to_draw.iter_rows(), save_dir=save_dir)


if __name__ == "__main__":
    visualize_objects(batch_predict())
