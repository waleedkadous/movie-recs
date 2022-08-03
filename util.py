import numpy as np
import os
from PIL import Image, ImageColor, ImageDraw, ImageFont
from typing import Union, Dict
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models.detection.ssd import ssd300_vgg16



import ray
from ray.train.torch import TorchPredictor


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

SCORE_THRESHOLD = 0.8
MARGIN = 4
WIDTH = 5
STROKE_WIDTH = 10


class VGG16Predictor(TorchPredictor):
    def call_model(self, tensor):
        model_output = super().call_model(tensor)

        # Find out the largest number of boxes, any of these
        # images have.
        pad_dim = max([output["boxes"].shape[0] for output in model_output])

        objects = {}
        for obj in model_output:
            for k, v in obj.items():
                if k not in objects: objects[k] = []
                v = v.detach().cpu().numpy()
                # Potentially pad the data for this column to the max length.
                pad = np.zeros((pad_dim - v.shape[0],) + v.shape[1:])
                v = np.concatenate((v, pad), axis=0)
                # Append so we can batch later.
                objects[k].append(v)

        for k, v in objects.items():
            objects[k] = torch.tensor(np.array(v))

        return objects

class SSDPredictor(TorchPredictor):
    def call_model(
        self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], pd.DataFrame]:
        """User predictor output formatting code."""
        model_output = super().call_model(tensor)
        return pd.DataFrame([
            {k: v.detach().cpu().numpy() for k, v in objects.items()}
            for objects in model_output
        ])
    

def convert_to_tensor(df: pd.DataFrame) -> pd.DataFrame:
    """User Pytorch code to transform user image."""
    preprocess = transforms.Compose(
        [transforms.ToTensor()]
    )
    df.loc[:, "image"] = [
        preprocess(np.asarray(image)).numpy() for image in df["image"]
    ]
    return df
    
def draw_bounding_boxes(df: dict) -> None:
    # Draw image first.
    image = (df["image"] * 255).astype("uint8")
    ndarr = np.transpose(image, (1, 2, 0))
    img_to_draw = Image.fromarray(ndarr)
    # Bounding box data.
    boxes = df["boxes"]
    labels = df["labels"].astype("uint8")
    scores = df["scores"]

    # Only keep high scoring boxes.
    boxes = boxes[scores > SCORE_THRESHOLD]
    labels = labels[scores > SCORE_THRESHOLD]
    str_labels = COCO_INSTANCE_CATEGORY_NAMES[(labels,)]

    num_boxes = boxes.shape[0]
    if num_boxes > 0:
        draw = ImageDraw.Draw(img_to_draw)

        colors = ["blue"] * num_boxes
        colors = [ImageColor.getrgb(color) for color in colors]
        txt_font = ImageFont.load_default()

        img_boxes = boxes.tolist()
        for bbox, color, label in zip(img_boxes, colors, str_labels):
            draw.rectangle(bbox, width=WIDTH, outline=color)
            draw.text(
                xy=(bbox[0] + MARGIN, bbox[1] + MARGIN),
                text=str(label),
                fill=color,
                stroke_width=STROKE_WIDTH,
            )

    return img_to_draw

def save_images(images_iter, save_dir: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, record in enumerate(images_iter):
        record.save(os.path.join(save_dir, f"img_{str(i).zfill(5)}.png"))

def visualize_objects(prediction_outputs: ray.data.Dataset):
    save_dir = "./object_detections"
    # Let's visualize the first 100 movie posters.
    images_to_draw = prediction_outputs.limit(100).map(draw_bounding_boxes)
    save_images(images_to_draw.iter_rows(), save_dir=save_dir)