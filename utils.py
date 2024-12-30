import config
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import torch
import torchvision.transforms as T


#################### GET IMAGE ID ####################
def get_image_id(image_path, coco):
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
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

    image_id = get_image_id(image_path, config.coco)

    annotations = get_image_annotations(image_id, config.coco)
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
def detect_objects(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path)
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])

    return prediction[0]


#################### ROUND BOX ####################
def round_box(bbox, tile_length):
    x1, y1, x2, y2 = bbox

    def round_to_nearest(value):
        lower_bound = (value // tile_length) * tile_length
        upper_bound = np.ceil(value / tile_length) * tile_length

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


#################### GET CATEGORY BY ID ####################
def get_category_by_id(categories, category_id):
    for category in categories:
        if category['id'] == category_id:
            return category['name']
    print(f'Category name for category ID {category_id} not found')
    return None


#################### GET BOX INDICES ####################
def get_box_indices(box, tile_length, num_rows, num_cols):
    x, y, x2, y2 = box
    width = x2 - x
    height = y2 - y

    col_idx = x // tile_length
    row_idx = y // tile_length

    width_tiles = width // tile_length
    height_tiles = height // tile_length

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


#################### GET CATEGORY NAME ####################
def get_category_name(category_id, coco):
    for _, category in coco.cats.items():
        if category['id'] == category_id:
            return category['name']
    print(f'Category name for category ID {category_id} not found')
    return None


#################### FILTER PREDICTIONS ####################
def filter_predictions(predictions, coco, verbose=False):
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
    SCORE_THRESHOLD = 0.8
    filtered_data = [item for item in filtered_data if item['score'] >= SCORE_THRESHOLD]

    if verbose:
        print('Number of detected objects after filtering:', len(filtered_data))

    # Iterate through remaining items and add category name
    for item in filtered_data:
        item['category'] = get_category_name(item['label'], coco)
        if verbose:
            print('Location:', item['box'])
            print('Category:', item['category'])
            print('Score:', item['score'])
            print()

    return filtered_data