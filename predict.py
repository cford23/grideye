import config
import matplotlib.pyplot as plt
from model import Model
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import utils


image_filename = 'f36e6c12-5ytnuhwh.png'
image_path = os.path.join(config.DATA_DIR, 'images', image_filename)

print('Image with correct boxes')
utils.display_original_image(image_path)

model = Model(num_classes=config.NUM_CATEGORIES)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

prediction = utils.detect_objects(model, image_path)
print('Image with predicted boxes')
utils.display_objects(image_path, prediction)

print('Image with rounded prediction boxes')
utils.display_objects(image_path, prediction, round_pred=True)

# TODO: Move the following into a function that can be used multiple times

# Filter model object predictions
filtered_preds = utils.filter_predictions(prediction, config.coco, verbose=False)

# Display filtered objects on original image
print('Image with filtered rounded prediction boxes')
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

color = 'red'
for det_object in filtered_preds:
    draw.rectangle(det_object['box'], outline=color, width=3)
    category_name = utils.get_category_name(det_object['label'], config.coco)
    score = det_object['score']
    draw.text(
        (det_object['box'][0], det_object['box'][1]),
        f'Class: {category_name}, Score: {score:.2f}',
        fill=color,
        font=font
    )

plt.imshow(image)
plt.axis('off')
plt.show()