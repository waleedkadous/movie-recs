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
    dataset = dataset.limit(int(0.25*dataset.count()))
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
    results = predictor.predict(dataset, num_gpus_per_worker=1, batch_size=128, keep_columns=["image"])
    return results
    
    
def visualize_objects(prediction_outputs: ray.data.Dataset):

    score_threshold = 0.8

    save_dir = "./object_detections"

    COCO_INSTANCE_CATEGORY_NAMES = np.array([
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ])

    def draw_bounding_boxes(
        df
    ) -> None:

        # TODO: Fix this once TensorArray auto-casting is resolved.
        image = df["image"] if type(df["image"]) == np.ndarray else df["image"].to_numpy()
        image = (image*255).astype("uint8")
        boxes = df["boxes"] if type(df["boxes"]) == np.ndarray else df["boxes"].to_numpy()
        labels = df["labels"] if type(df["labels"]) == np.ndarray else df["labels"].to_numpy()
        scores = df["scores"] if type(df["scores"]) == np.ndarray else df["scores"].to_numpy()


        # Only keep high scoring boxes.
        boxes = boxes[scores > score_threshold]
        labels = labels[scores > score_threshold]
        str_labels = COCO_INSTANCE_CATEGORY_NAMES[[labels]]

        num_boxes = boxes.shape[0]


        colors = ["blue"] * num_boxes

        colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

        txt_font = ImageFont.load_default()

        ndarr = np.transpose(image, (1, 2, 0))
        img_to_draw = Image.fromarray(ndarr)
        img_boxes = boxes.tolist()

        draw = ImageDraw.Draw(img_to_draw)

        for bbox, color, label in zip(img_boxes, colors, str_labels):  # type: ignore[arg-type]
            draw.rectangle(bbox, width=5, outline=color)

            margin = 4
            draw.text(xy=(bbox[0]+margin, bbox[1]+margin), text=str(label), fill=color, stroke_width=10)


        return img_to_draw


    # Let's visualize the first 100 movie posters.
    images_to_draw = first_results_window.limit(100).map(draw_bounding_boxes)

    save_dir = "./object_detections"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, record in enumerate(images_to_draw.iter_rows()):
        record.save(os.path.join(save_dir, "img_"+str(i).zfill(5)+".png"))
        
        
if __name__ == "__main__":
    dataset = read_data_from_s3()
    prediction_results = batch_predict(dataset)
    visualize_objects(prediction_results)

