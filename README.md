# movie-recs
Movie poster object detection with Ray and Anyscale

The following workload has been run on a ml.g4dn.12xlarge Sagemaker instance.

# Requirements
```
pip install ray[all]
ray install-nightly
pip install torch
pip install torchvision
pip install tqdm # to visualize progress bar
```

# PyTorch Serial
First download the images from s3
```
mkdir images
aws s3 cp --recursive s3://waleed-movies ./images/
```

Then run `python pytorch_serial.py`

# Running Ray locally on Sagemaker
`python movie_poster_rec.py`

This will
1. Read the Dataset from S3
2. Perform batch prediction with Ray AIR
3. Take the first handful of prediction results to visualize in the `./object_detections` folder.

# Running Ray on Anyscale via Anyscale Connect
TODO

# Running Ray on Anyscale via Workspaces
TODO
