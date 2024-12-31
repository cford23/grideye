import logging
import os
from pycocotools import coco


DATA_DIR = 'data/'
COCO = coco.COCO(os.path.join(DATA_DIR, 'result.json'))
NUM_CATEGORIES = len(COCO.cats) + 1 # for background

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

# Filter score threshold for bounding boxes
SCORE_THRESHOLD = 0.8

# Integer to be converted to from box label
SQUARE_CODES = {
    0: ['house', 'ledge boundary', 'pokecenter', 'pokemart', 'tree'],
    2: ['npc'],
    4: ['ledge'],
    5: ['player'],
    6: ['grass'],
    7: ['sign']
}