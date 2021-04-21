# DVC Example: Versionning and distributing TFRecords and Keras saved models 

This example demonstrates the use of DVC to implement a basic model training workflow. Particularly, it's used to achieve the following:
- Transfer data to remote servers to run model training
- Save serialized models following training

**Ensure you have all the python packages listed in `requirements.txt` installed. We also recommend that you download the contents of this repository without `.git` so that you can initialize it and link it to your own remote from scratch.**

***Note**: The data loading, data transformation, and model training code in this example is taken from: https://keras.io/examples/keras_recipes/creating_tfrecords*


## 1. Download, extract, and convert training data to TFRecords:

Let's start by fetching and decompressing our training data. To do this, move into the `dvc-gdrive-example` directory and run the following commands:
```bash
mkdir -p data/raw # note that this subdirectory is ignored in .gitignore

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/raw/annotations.zip
unzip data/raw/annotations.zip -d data/raw && rm data/raw/annotations.zip

wget http://images.cocodataset.org/zips/val2017.zip -O data/raw/images.zip
unzip data/raw/images.zip -d data/raw && mv data/raw/val2017 data/raw/images && rm data/raw/images.zip
```

Now that the raw data is there, TFRecords can be generated using our transform script:
```bash
mkdir data/tfrecords # this subdirectory is also git-ignored
python transform.py # only 18% of the dataset is converted to TFRecords to speed-up the demo
```


## 2. Initialize, authenticate, and push data to DVC:

With our data ready to be distributed, we initialize DVC:
```bash
dvc init # note the creation of a `.dvc` directory
```

DVC can use multiple backends to store the data such as S3 buckets, GCS buckets, NFS, Google Drive. For the sake of simplicity, we'll use the latter.

Create a folder in your drive and move into it. The URL in your browser should be something like:
```
https://drive.google.com/drive/u/2/folders/1zqYOkK4GxoyS5xDwqzqjYlmXlyWCNU0_
```

We're interested in the folder ID, which is the last part of the URL. We use it to create a DVC Google Drive remote:
```bash
dvc remote add -d storage gdrive://1zqYOkK4GxoyS5xDwqzqjYlmXlyWCNU0_
git add .dvc/config
git commit -m "updated DVC config"
```

Now we try tracking our TFRecords in the git repository with DVC then push them to the remote in preparation for training:
```bash
dvc add data/tfrecords
git add data/tfrecords.dvc data/.gitignore
git commit -m "added DVC metadata for data/tfrecords"
dvc push # because we use Google Drive in its most basic protocols, a browser sign-in is required
git push # you'll have to setup a remote git repository to do this
```


## 3. Load data using DVC and launch training:

We're ready to fetch the data and launch training in a remote server. For this demo, our remote, isolated environment is nothing more than a Docker container:

```bash
docker build -t dvc-gdrive-example . 
docker run -it -v $HOME/.ssh:/root/.ssh dvc-gdrive-example bash # we inject our ssh keys to enable authenticating with Github
```
***Note:**: In production-grade environments, Git remote authentication is done using access tokens or by generating dedicated SSH/GPG keys, unlike this demo where we inject our development credentions to the training environment (container).*

The container image contains the git repository and our dependencies installed. Now we fetch the data and launch training:
```bash
dvc pull # authentication is required again
python train.py $(git rev-parse --short HEAD) # model is saved to models/checkpoints/ which is git-ignored
```
***Note:**: In production-grade workloads, DVC remote authentication is automated by generating credential files and injecting them to the training environment for DVC to use without requiring interactive action like we do in this demo.*

Finally, we save the trained model checkpoint:
```bash
dvc add models/checkpoints
git add models/checkpoints.dvc models/.gitignore
git commit -m "added DVC metadata for model checkpoint $(git rev-parse --short HEAD)"
dvc push
git push
```

## 4. Verify training results in local environment:

After running the training workload, the results should be available for us to pull into our local environment (host):
```bash
git pull
dvc pull
```