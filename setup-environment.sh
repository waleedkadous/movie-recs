#!/bin/sh
# Install the appropriate versions of things 
pip install https://ray-ci-artifact-pr-public.s3.amazonaws.com/c58874ae8545eef0b5c7632418eba3da0b5015c9/tmp/artifacts/.whl/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
pip install ray[tune]
pip install torch
pip install torchvision
pip install tqdm
