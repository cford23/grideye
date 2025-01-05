try:
    # Attempt relative import (works when run as part of a package)
    from . import config
except ImportError:
    # Fallback to absolute import (works when run independently)
    import config

import argparse
import os
from PIL import Image
import utils
import screen
import random
import statistics


def main():
    # Set up commandline arguments
    parser = argparse.ArgumentParser(description='Use GRIDEYE object detection model to get environment state from emulator screenshot.')
    parser.add_argument(
        '--image', 
        type=str, 
        help='Filepath for image to make prediction on.'
    )
    args = parser.parse_args()

    # Set image from commandline argument
    if args.image is None:
        # No image provided - take screenshot of emulator
        image = screen.take_screenshot('Pokemon - Emerald Version')
        screenshots_path = os.path.join(os.getcwd(), 'data', 'screenshots')
        if not os.path.isdir(screenshots_path):
            os.mkdir(screenshots_path)

        random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        image_filename = f'{random_string}.png'
        image_path = os.path.join(screenshots_path, image_filename)

        # Save screenshot
        image.save(image_path)
    else:
        # Image provided - load given image
        image_path = os.path.join(config.DATA_DIR, args.image)
        image = Image.open(image_path)

        # Display image with correct boxes
        utils.display_original_image(image_path, title='Image with correct boxes')

    # Get object predictions and organize into correct format
    model = utils.load_model()
    predictions = utils.detect_objects(model, image)
    predictions = utils.organize_predictions(predictions)

    label_scores = {}
    # Get scores per label
    for prediction in predictions:
        category = prediction['category']
        if category not in label_scores:
            label_scores[category] = []
        label_scores[category].append(prediction['score'])

    for key, value in label_scores.items():
        print('Label:', key)
        print('Scores:', value)
        print('Average Score:', statistics.mean(value))
        print('Median Score:', statistics.median(value))
        print('Max Score:', max(value))
        print('Min Score:', min(value))
        print()

    # Display image with predicted boxes
    utils.display_image(image_path, predictions, title='Image with predicted boxes')

    # Display image with filtered prediction boxes
    filtered_preds = utils.filter_predictions(predictions)
    utils.display_image(image_path, filtered_preds, title='Image with filtered prediction boxes')


if __name__ == '__main__':
    main()
