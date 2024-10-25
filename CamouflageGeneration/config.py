import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import os
# Mac book
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

TRAIN_DIR = "/Users/manas/Documents/GitHub/Cycle-GAN/CamouflageGeneration/dataset/horsezebra"
VAL_DIR = "/Users/manas/Documents/GitHub/Cycle-GAN/CamouflageGeneration/dataset/horsezebra"
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"
LAMBDA_COLOR = 10

time = str(time.time())
RESULTS = "results/" + time

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)