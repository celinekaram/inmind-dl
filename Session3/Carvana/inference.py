from model import UNET
from utils import (
    load_checkpoint,
    save_predictions_as_imgs
)
# save weights in a file for inference

def main():
    load_checkpoint(checkpoint, UNET)
    