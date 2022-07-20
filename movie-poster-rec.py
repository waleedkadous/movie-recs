files_dir = "/home/ec2-user/SageMaker/movie-recs/movie-posters"

import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import json
import ray
from ray.air.util.tensor_extensions.pandas import TensorArray
from ray.train.torch import to_air_checkpoint, TorchPredictor
from ray.train.batch_predictor import BatchPredictor

from torchvision import transforms
from torchvision.models.detection.ssd import ssd300_vgg16

ray.init("anyscale://movie-recs",
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/home/ec2-user/SageMaker/object-spill"}},
        )
    },
)

def convert_to_pandas(byte_item_list):
    preprocess = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop((300,600)),
        transforms.ToTensor(),
    ])

    images = [Image.open(BytesIO(byte_item)).convert('RGB') for byte_item in byte_item_list]
    images = [preprocess(image) for image in images]
    images = [np.array(image) for image in images]

    return pd.DataFrame({"image": TensorArray(images)})

# What is parallelism magic number?
dataset = ray.data.read_binary_files(paths=files_dir, parallelism=2000)
dataset = dataset.map_batches(convert_to_pandas)


model = ssd300_vgg16(pretrained=True)

# # TODO: to_air_checkpoint should accept a model or a model state dict
ckpt = to_air_checkpoint(model=model)

predictor = BatchPredictor.from_checkpoint(ckpt, TorchPredictor)
results = predictor.predict(dataset, num_gpus_per_worker=1)
pickle.dump(results, open('results.pickle', 'wb'))