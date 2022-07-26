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
3. Take the first 100 prediction results to visualize in the `./object_detections` folder.

# Running Ray on Anyscale via Workspaces
1. Set up Anyscale git access so you can clone the repo. 
	- You can either create a [Personal Access token on Github](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). 
	- Or you can generate an ssh key on the Sagemaker instance and add it to your Github account 
2. Configure the Personal Access Token/SSH key with Anyscale SSO
3. Clone the product repo `git clone git@github.com:anyscale/product.git`
4. Install Anyscale CLI
```
cd product/frontend/cli
pip install -e .
```
4. Create a workspace on Anyscale staging. Use the `sagemaker-cluster-env:1` cluster env. Use 4 `g4dn.12xlarge` nodes.
5. Add Anyscale credentials to your Sagemaker instance
Go to https://console.anyscale-staging.com/o/anyscale-internal/credentials and follow the instructions for setting environment variables. Seems like `anyscale auth` does not work for staging.
For example
```
export ANYSCALE_HOST=https://console.anyscale-staging.com
export ANYSCALE_CLI_TOKEN=<INSERT_YOUR_CLI_TOKEN>
```
6. Clone the workspace in your Sagemaker Notebook. For example, if your workspace is called `sagemaker-demo`, then do `anyscale workspace clone -n sagemaker-demo`
7. Copy files to your workspace `cp *.py workspace-project-sagemaker-demo`
8. Run batch prediction
```
cd workspace-project-sagemaker-demo
anyscale workspace run "python movie-poster-rec.py"
```
9. If you access your workspace from Anyscale, you should see the first 100 object detections saved on the head node.

# Running Ray on Anyscale via Anyscale Connect
Follow instructions 1-5 from the previous section.

Now do the following
1. Add the appropriate Ray address to your script. 
For example, add the following line to `movie_poster_rec.py`: `ray.init("anyscale://workspace-project-sagemaker-demo/workspace-cluster-sagemaker-demo", runtime_env={"working_dir": "."})`
2. Run `python movie_poster_rec.py`
3. You should see the 100 object detections saved on the locally on the Sagemaker notebook.
