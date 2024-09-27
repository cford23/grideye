# Converts data in object_locations.csv to COCO format for object detection model
from datetime import datetime
import json
import os
import pandas as pd
from PIL import Image


NUM_SCREEN_ROWS = 9
NUM_SCREEN_COLS = 15


def get_category_id(categories, name):
    for category in categories:
        if category['name'] == name:
            return category['id']
    return None


# Change to /data
os.chdir('../grideye-object-detection/data')

# Load CSV with object locations
df = pd.read_csv('object_locations.csv')

current_date = datetime.now()
current_date_str = current_date.strftime('%Y-%m-%d')

# Initialize data for COCO format
# https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html#md-coco-images
data = {
    'info': {
        "description": "Pokemon Emerald screenshots",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": int(current_date.strftime('%Y')),
        "contributor": "COCO Consortium",
        "date_created": current_date_str
    },
    "licenses": [{ "url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License" }],
    "images": [],
    "annotations": [],
    "categories": [
        { 'supercategory': 'boundary', 'id': 0, 'name': 'tree' },
        { 'supercategory': 'boundary', 'id': 1, 'name': 'sign' },
        { 'supercategory': 'boundary', 'id': 2, 'name': 'ledge boundary' },

        { 'supercategory': 'building', 'id': 3, 'name': 'pokecenter' },
        { 'supercategory': 'building', 'id': 4, 'name': 'pokemart' },
        { 'supercategory': 'building', 'id': 5, 'name': 'house' },
        { 'supercategory': 'building', 'id': 6, 'name': 'building' },

        { 'id': 7, 'name': 'open' },
        { 'id': 8, 'name': 'door' },
        
        { 'supercategory': 'character', 'id': 9, 'name': 'npc' },
        { 'supercategory': 'character', 'id': 10, 'name': 'player' },
        
        { 'id': 11, 'name': 'ledge' },
    ]
}

image_name = None
image = None
tile_length = None
image_id = 0
annot_count = 1
new_image_size = (1440, 864)
for index, row in df.iterrows():
    if row['image'] != image_name:
        image_id += 1
        # Add new item to images list
        image_name = row['image']
        image_path = os.path.join('images', image_name)
        image = Image.open(image_path)
        image = image.resize(new_image_size)
        tile_length = image.size[1] // NUM_SCREEN_ROWS
        assert tile_length == image.size[0] // NUM_SCREEN_COLS, 'Tile lengths do not match'

        current_datetime = datetime.now()
        data['images'].append({
            'id': image_id,
            'width': image.size[0],
            'height': image.size[1],
            'file_name': image_name,
            'date_captured': current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        })

    # Get row and column indices from tile number
    tile_num = row['top_left_tile_number']
    row_index = (tile_num - 1) // NUM_SCREEN_COLS
    col_index = (tile_num - 1) % NUM_SCREEN_COLS

    
    category_id = get_category_id(data['categories'], row['label'])
    if category_id is None:
        print('Category ID not found for label: ', row['label'])
        exit()

    # Get x_min and y_min from row and column indices
    x_min = col_index * tile_length
    y_min = row_index * tile_length
    x_max = x_min + (tile_length * row['width_tiles'])
    y_max = y_min + (tile_length * row['height_tiles'])

    data['annotations'].append({
        'id': annot_count,
        'image_id': image_id,
        'category_id': category_id,
        'area': (x_max - x_min) * (y_max - y_min),
        'bbox': [x_min, y_min, x_max, y_max]
    })
    annot_count += 1

with open('annotations.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
