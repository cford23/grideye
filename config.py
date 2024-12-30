import logging
import os
from pycocotools.coco import COCO


DATA_DIR = 'data/'
coco = COCO(os.path.join(DATA_DIR, 'result.json'))
NUM_CATEGORIES = len(coco.cats) + 1 # for background

# Image details
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 432
NUM_SCREEN_ROWS = 9
NUM_SCREEN_COLS = 15
TILE_LENGTH = IMAGE_HEIGHT // NUM_SCREEN_ROWS

# Trainer parameters
EPOCHS = 100
ACCELERATOR = 'auto'
DEVICES = 1
CHECKPOINTING = False
LOG_N_STEPS = 1

# Hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Saved model path
MODEL_PATH = os.path.join(DATA_DIR, 'grideye_model.pth')