# movie-recs
Movie poster object detection with Ray and Anyscale

The following workload has been run on a ml.g4dn.12xlarge Sagemaker instance.

# Requirements
```
pip install ray[all]
ray install-nightly
pip install torch
pip install torchvision
```

# PyTorch Serial
TODO

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
