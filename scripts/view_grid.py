import argparse
import os
from PIL import Image, ImageDraw, ImageFont


NUM_SCREEN_ROWS = 9
NUM_SCREEN_COLS = 15

parser = argparse.ArgumentParser(description='Displays image with grid and numbers for data labeling')
parser.add_argument('filename', type=str, help='Image file to add grid to')
args = parser.parse_args()

os.chdir('../data')
image_path = os.path.join('images', args.filename)
image = Image.open(image_path)

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

tile_length = image.size[1] // NUM_SCREEN_ROWS
assert tile_length == image.size[0] // NUM_SCREEN_COLS, 'Tile lengths do not match'

# Add grid to image
width, height = image.size
# Draw vertical lines
for x in range(0, width, tile_length):
    draw.line([(x, 0), (x, height)], fill='black', width=1)

# Draw horizontal lines
for y in range(0, height, tile_length):
    draw.line([(0, y), (width, y)], fill='black', width=1)

# Add tile number to image
count = 1
for row in range(NUM_SCREEN_ROWS):
    for col in range(NUM_SCREEN_COLS):
        x_min = col * tile_length
        y_min = row * tile_length
        
        text = str(count)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x_min + (tile_length - text_width) // 2
        text_y = y_min + (tile_length - text_height) // 2

        draw.text((text_x, text_y), text, fill='black', font=font)
        count += 1

image.show()