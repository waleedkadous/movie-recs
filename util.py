import numpy as np
import pandas as df

import os

from PIL import Image, ImageColor, ImageDraw, ImageFont

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

def draw_bounding_boxes(df: dict) -> None:
    score_threshold = 0.8

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

def save_images(images_iter, save_dir: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, record in enumerate(images_iter):
        record.save(os.path.join(save_dir, "img_"+str(i).zfill(5)+".png"))
