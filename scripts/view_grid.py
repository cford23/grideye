import argparse
import config
import os
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser(description='Displays image with grid and numbers for data labeling')
parser.add_argument('filename', type=str, help='Image file to add grid to')
args = parser.parse_args()

os.chdir('../data')
image_path = os.path.join('images', args.filename)
image = Image.open(image_path)

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Add grid to image
# Draw vertical lines
for x in range(0, config.IMAGE_WIDTH, config.TILE_LENGTH):
    draw.line([(x, 0), (x, config.IMAGE_HEIGHT)], fill='black', width=1)

# Draw horizontal lines
for y in range(0, config.IMAGE_HEIGHT, config.TILE_LENGTH):
    draw.line([(0, y), (config.IMAGE_WIDTH, y)], fill='black', width=1)

# Add tile number to image
count = 1
for row in range(config.NUM_SCREEN_ROWS):
    for col in range(config.NUM_SCREEN_COLS):
        x_min = col * config.TILE_LENGTH
        y_min = row * config.TILE_LENGTH
        
        text = str(count)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x_min + (config.TILE_LENGTH - text_width) // 2
        text_y = y_min + (config.TILE_LENGTH - text_height) // 2

        draw.text((text_x, text_y), text, fill='black', font=font)
        count += 1

image.show()