import logging

DATA_DIR = 'data/'

# Trainer parameters
EPOCHS = 1
ACCELERATOR = 'auto'
DEVICES = 1
CHECKPOINTING = False
LOG_N_STEPS = 1

# Hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)