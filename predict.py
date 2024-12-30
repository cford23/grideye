import config
from model import Model
import os
import torch
import utils


image_path = os.path.join(config.DATA_DIR, 'images', 'a816fba6-image_83.png')
coco_path = os.path.join(config.DATA_DIR, 'result.json')
print('Image with correct boxes')
utils.display_original_image(image_path, coco_path)

model = Model(num_classes=config.NUM_CATEGORIES)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

prediction = utils.detect_objects(model, image_path)
print('Image with predicted boxes')
utils.display_objects(image_path, prediction)

print('Image with rounded prediction boxes')
utils.display_objects(image_path, prediction, round_pred=True)