import config
from functools import wraps
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import torch
import torchvision.transforms as T


#################### GET IMAGE ANNOTATIONS ####################
def get_image_annotations(image_filename):
    img_ids = config.coco.getImgIds()
    img_id = None
    for img_id in img_ids:
        img_info = config.coco.imgs[img_id]
        if image_filename in img_info['file_name']:
            break
    else:
        print(f"Image '{image_filename}' not found.")
        return []

    print('Image ID:', img_id)

    ann_ids = config.coco.getAnnIds(imgIds=img_id)
    annotations = config.coco.loadAnns(ann_ids)

    print('Number of annotations:', len(annotations))

    return annotations


#################### DISPLAY ORIGINAL IMAGE ####################
def display_original_image(image_path):
    image_filename = image_path.split('/')[-1]

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    annotations = get_image_annotations(image_filename, config.coco)
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
