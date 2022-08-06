#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Full dataset 41000 images

# FILES_DIR = "/home/ec2-user/SageMaker/movie_posters/img_41K"

## CHANGE 1: Read files from S3 instead of locally
FILES_DIR = "s3://waleed-movies"

# CHANGE 2: Point at a cluster 
get_ipython().run_line_magic('env', 'ANYSCALE_HOST=https://console.anyscale-staging.com')
CLUSTER_URL = "anyscale://workspace-project-demo/workspace-cluster-demo"
RUNTIME = {"working_dir": ".", "env_vars": {"RAY_SCHEDULER_EVENTS": "0"}}

# CHANGE 3: Increase workers from 4 to 20. 
NUM_WORKERS=80

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import json

import ray
from ray.train.torch import TorchPredictor, TorchCheckpoint
from ray.train.batch_predictor import BatchPredictor
from ray.data.preprocessors import BatchMapper
from ray.data.datasource import ImageFolderDatasource
import anyscale

from torchvision.models.detection.ssd import ssd300_vgg16

from util import visualize_objects, convert_to_tensor, SSDPredictor

# TODO: Enable auto casting once we resolve call_model() output format
from ray.data.context import DatasetContext
ctx = DatasetContext.get_current();
ctx.enable_tensor_extension_casting = False


# In[2]:


def batch_predict(files_dir):
    dataset = ray.data.read_datasource(
        ImageFolderDatasource(), root=files_dir, size=(300, 300), mode="RGB"
    )
    preprocessor = BatchMapper(convert_to_tensor)
    model = ssd300_vgg16(pretrained=True)
    ckpt = TorchCheckpoint.from_model(model=model, preprocessor=preprocessor)
    predictor = BatchPredictor.from_checkpoint(ckpt, SSDPredictor)

    return predictor.predict(dataset, 
                             batch_size=128,
                             min_scoring_workers=NUM_WORKERS,
                             max_scoring_workers=NUM_WORKERS, 
                             num_cpus_per_worker=4, 
                             num_gpus_per_worker=1, 
                             feature_columns=["image"], 
                             keep_columns=["image"])




# In[3]:


ray.init(CLUSTER_URL, runtime_env=RUNTIME)


# In[4]:


prediction_results = batch_predict(FILES_DIR)


# In[ ]:




