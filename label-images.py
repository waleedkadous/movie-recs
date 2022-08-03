#!/usr/bin/env python
# coding: utf-8

# In[15]:


# First install everything -- you only need to do this once: 
#! pip install https://ray-ci-artifact-pr-public.s3.amazonaws.com/c58874ae8545eef0b5c7632418eba3da0b5015c9/tmp/artifacts/.whl/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
#! pip install ray[tune]
#! pip install torch
#! pip install torchvision
#! pip install tqdm
# ! pip install anyscale
# %env ANYSCALE_HOST=https://console.anyscale-staging.com
# %env ANYSCALE_CLI_TOKEN=sss_4nneyStJk8ORsSgxW45lDT
# %env IGNORE_VERSION_CHECK=1


# In[16]:


FILES_DIR = "s3://waleed-movies"

# files_dir = "/home/ec2-user/images"

# Use this for quick testing with ~300 files
# FILES_DIR = "s3://air-example-data-2/movie-image-small-filesize-1-file"
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


# In[17]:


def batch_predict(files_dir, lim):
    dataset = ray.data.read_datasource(
        ImageFolderDatasource(), root=files_dir, size=(300, 300), mode="RGB"
    ).limit(lim)

    preprocessor = BatchMapper(convert_to_tensor)
    model = ssd300_vgg16(pretrained=True)
    ckpt = TorchCheckpoint.from_model(model=model, preprocessor=preprocessor)
    predictor = BatchPredictor.from_checkpoint(ckpt, SSDPredictor)
    return predictor.predict(dataset, 
                             batch_size=128,
                             min_scoring_workers=20,
                             max_scoring_workers=20, 
                             num_cpus_per_worker=4, 
                             num_gpus_per_worker=1, 
                             feature_columns=["image"], 
                             keep_columns=["image"])


# In[8]:


ray.init("anyscale://workspace-project-demo/workspace-cluster-demo", runtime_env={"working_dir": "."})


# In[11]:


prediction_results = batch_predict(FILES_DIR, 100000)


# In[12]:


prediction_results


# In[14]:





# In[ ]:




