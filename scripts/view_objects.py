# Takes an image and displays it with rectangles over detected objects
import argparse
import json
import os
from PIL import Image, ImageDraw


def get_image(images, image_name):
    for image in images:
        if image['file_name'] == image_name:
            return image
    return None


def get_annotations(annotations, image_id):
    image_annotations = []
    for object in annotations:
        if object['image_id'] == image_id:
            image_annotations.append(object)
    return image_annotations


def main():
    os.chdir('../data')
    with open('annotations.json', 'r') as json_file:
        data = json.load(json_file)
    
    parser = argparse.ArgumentParser(description='Displays image with identified objects from labeled data')
    parser.add_argument('filename', type=str, help='Image file to display objects from')
    args = parser.parse_args()

    image = get_image(data['images'], args.filename)
    annotations = get_annotations(data['annotations'], image['id'])
    image_size = (image['width'], image['height'])

    image_path = os.path.join('images', args.filename)
    image = Image.open(image_path)
    image = image.resize(image_size)
    draw = ImageDraw.Draw(image)

    for object in annotations:
        draw.rectangle(object['bbox'], outline="red", width=3)

    image.show()


if __name__ == '__main__':
    main()
