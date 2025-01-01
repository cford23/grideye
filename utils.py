from . import config
from functools import wraps
import matplotlib.pyplot as plt
from .model import Model
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import torch
import torchvision.transforms as T


#################### GET IMAGE ID ####################
def get_image_id(image_path):
    for image_id in config.COCO.getImgIds():
        image_info = config.COCO.loadImgs(image_id)[0]
        if image_info['file_name'] in image_path:
            return image_info['id']
    return None


#################### GET IMAGE ANNOTATIONS ####################
def get_image_annotations(image_id, coco):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    return annotations


#################### DISPLAY ORIGINAL IMAGE ####################
def display_original_image(image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    image_id = get_image_id(image_path, config.COCO)

    annotations = get_image_annotations(image_id, config.COCO)
    print('Number of annotations:', len(annotations))
    for ann in annotations:
        x, y, width, height = ann['bbox']
        draw.rectangle([x, y, x + width, y + height], outline='red', width=3)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


#################### TIME IT ####################
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result
    return wrapper


#################### DETECT OBJECTS ####################
@timeit
def detect_objects(model: Model, image: Image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])

    return prediction[0]


#################### ROUND BOX ####################
def round_box(bbox):
    x1, y1, x2, y2 = bbox

    def round_to_nearest(value):
        lower_bound = (value // config.TILE_LENGTH) * config.TILE_LENGTH
        upper_bound = np.ceil(value / config.TILE_LENGTH) * config.TILE_LENGTH

        # Check which is closer
        if abs(value - lower_bound) <= abs(value - upper_bound):
            return int(lower_bound)  # Closer to lower bound
        else:
            return int(upper_bound)  # Closer to upper bound

    x1_rounded = round_to_nearest(x1)
    y1_rounded = round_to_nearest(y1)
    x2_rounded = round_to_nearest(x2)
    y2_rounded = round_to_nearest(y2)

    return [x1_rounded, y1_rounded, x2_rounded, y2_rounded]


#################### GET BOX INDICES ####################
def get_box_indices(box):
    x, y, x2, y2 = box
    width = x2 - x
    height = y2 - y

    col_idx = x // config.TILE_LENGTH
    row_idx = y // config.TILE_LENGTH

    width_tiles = width // config.TILE_LENGTH
    height_tiles = height // config.TILE_LENGTH

    indices_in_box = [(row, col) for row in range(row_idx, row_idx + height_tiles)
                                 for col in range(col_idx, col_idx + width_tiles)]

    return indices_in_box


#################### DISPLAY OBJECTS ####################
def display_objects(image_path, prediction, round_pred=False, verbose=False, draw_grid=False):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if draw_grid:
        # Draw vertical lines
        for x in range(0, config.IMAGE_WIDTH, config.TILE_LENGTH):
            draw.line([(x, 0), (x, config.IMAGE_HEIGHT)], fill='black', width=1)

        # Draw horizontal lines
        for y in range(0, config.IMAGE_HEIGHT, config.TILE_LENGTH):
            draw.line([(0, y), (config.IMAGE_WIDTH, y)], fill='black', width=1)

    color = 'red'
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        box = [round(i.item()) for i in box.cpu()]
        if round_pred:
            box = round_box(box, config.TILE_LENGTH)
        label = label.item()
        score = score.item()
        if verbose:
            print('Box:', box)
            print('Label:', label)
            print('Score:', score)
            print()

        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), f'Class: {label}, Score: {score:.2f}', fill=color, font=font)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


#################### FILTER PREDICTIONS ####################
def filter_predictions(predictions, verbose=False):
    objects = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        box = [round(i.item()) for i in box.cpu()]
        box = round_box(box, config.TILE_LENGTH)
        objects.append({
            'box': box,
            'label': label.item(),
            'score': score.item()
        })

    # Removes any locations that have multiple objects predictions and keeps the highest scored object
    sorted_objects = sorted(objects, key=lambda x: (x['box'], -x['score']))
    unique_data = {}
    for item in sorted_objects:
        values = tuple(item['box'])  # Convert list to tuple to use as a dictionary key
        if values not in unique_data or item['score'] > unique_data[values]['score']:
            unique_data[values] = item

    # Extract the filtered list
    filtered_data = list(unique_data.values())

    # Filter out any low scoring objects
    filtered_data = [item for item in filtered_data if item['score'] >= config.SCORE_THRESHOLD]

    if verbose:
        print('Number of detected objects after filtering:', len(filtered_data))

    # Iterate through remaining items and add category name
    for item in filtered_data:
        # Get category name based on category ID
        item['category'] = config.COCO.cats[item['label']]['name']
        if verbose:
            print('Location:', item['box'])
            print('Category:', item['category'])
            print('Score:', item['score'])
            print()

    return filtered_data


#################### LOAD MODEL ####################
def load_model():
    model = Model(num_classes=config.NUM_CATEGORIES)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()
    return model


#################### GET ENVIRONMENT STATE ####################
def get_environment_state(model: Model, image: Image):
    # Get object detection predictions from given image
    predictions = detect_objects(model, image)

    # Filter the bounding box predictions
    filtered_predictions = filter_predictions(predictions)

    # For remaining predictions, convert to 2D array
    state = np.ones((config.NUM_SCREEN_ROWS, config.NUM_SCREEN_COLS))

    for item in filtered_predictions:
        # Get a list of indices that box covers
        box_indices = get_box_indices(item['box'])

        # For each index, change value from 1 to correct value
        square_code = next((k for k, v in config.SQUARE_CODES.items() if item['category'] in v), None)
        if square_code is None:
            print('Failed to find square code for', item['category'])
            break

        for idx in box_indices:
            row, col = idx
            state[row][col] = square_code

    # returns 2D array of integers representing the game's current state
    return state
