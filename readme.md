# Quick Start Guide

To run this project locally it is recommended to use a virtual environment (Conda or Pip).
The environment used for this project is defined in [requirements.txt](requirements.txt) or for conda specifically in [env.yml](env.yml).

## Setup

**Conda**
```bash
conda env create -f env.yml
conda activate video-captioning
```

**Pip**
```bash
pip install -r requirements.txt
```

## Usage

**Retrain**

1. Download the dataset [Microsoft Research Video Description Corpus (MSVD)](https://www.kaggle.com/datasets/vtrnanh/msvd-dataset-corpus) and extract it to the `data/train/videos` directory.
2. Run the script [preprocess.py](src/preprocess.py) to extract the features from the videos (took 3 hours for me)
3. Run the script [train.py](src/train.py) to train the model (took 3 hours for me)

**Inference**

- Run the script [inference.py](src/inference.py) to generate captions for a random selection of videos from the dataset
- View the results in `data/test/test_greedy.txt` or `data/test/test_beam_search.txt` respectively

**Realtime**

- Run the script [realtime.py](src/realtime.py) to interactively generate captions and display the video in real-time. For this the videos have to be located in the `data/realtime` directory.
