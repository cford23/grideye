import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import config
import os
from PIL import Image
import Quartz.CoreGraphics as CG
import random
import time


#################### FIND WINDOW BY NAME ####################
def find_window_by_name(window_name):
    window_list = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID)
    for window in window_list:
        if window_name.lower() in window.get('kCGWindowName', '').lower():
            return window
    return None


#################### CROP IMAGE ####################
def crop_image(image: Image):
    width, height = image.size

    tile_length = width // config.NUM_SCREEN_COLS

    # Remove half a tile off the bottom
    # Remove half a tile off the top plus the title bar
    window_title_height = height - tile_length - (tile_length * config.NUM_SCREEN_ROWS)

    # Get bounds of image to crop
    left = 0
    upper = window_title_height + (tile_length // 2)
    right = width
    lower = height - (tile_length // 2)
    box = (left, upper, right, lower)

    # Crop image from bounds
    image = image.crop(box)

    return image


#################### SCREENSHOW WINDOW ####################
def screenshot_window(window):
    '''Takes a screenshot of the given window and returns the correctly cropped image of it'''
    window_id = window['kCGWindowNumber']
    bounds = window['kCGWindowBounds']

    x = int(bounds['X'])
    y = int(bounds['Y'])
    width = int(bounds['Width'])
    height = int(bounds['Height'])
    
    # Create a screenshot of the specific window
    image_ref = CG.CGWindowListCreateImage(
        CG.CGRectMake(x, y, width, height),
        CG.kCGWindowListOptionIncludingWindow,
        window_id,
        CG.kCGWindowImageDefault
    )

    if not image_ref:
        print('Failed to capture screenshot')
        return None
    
    width = CG.CGImageGetWidth(image_ref)
    height = CG.CGImageGetHeight(image_ref)
    bytes_per_row = CG.CGImageGetBytesPerRow(image_ref)
    data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image_ref))

    # Convert the screenshot to a PIL image with the correct color mode
    image = Image.frombuffer('RGBA', (width, height), data, 'raw', 'BGRA', bytes_per_row, 1)
    image = image.convert('RGB')
    
    image = crop_image(image)

    # Resize image to IMAGE_WIDTH x IMAGE_HEIGHT
    image = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    return image


#################### MAIN ####################
def main():
    # Set up commandline arguments
    parser = argparse.ArgumentParser(description='Take screenshot periodically based on an optional argument until Ctrl-C.')
    parser.add_argument(
        '--interval', 
        type=int, 
        help='Time interval in seconds for taking the screenshot.'
    )
    args = parser.parse_args()

    # Make sure the /screenshots folder exists (if not create it)
    screenshots_path = os.path.join(os.getcwd(), 'data', 'screenshots')
    if not os.path.isdir(screenshots_path):
        os.mkdir(screenshots_path)

    # Set interval from commandline argument
    interval = args.interval
    if interval is not None:
        if interval <= 0:
            interval = None

    window_name = 'Pokémon: Emerald Version'
    try:
        while True:
            print('Taking screenshot...', end='', flush=True)
            window = find_window_by_name(window_name)
            if not window:
                print('No window found with the name:', window_name)
                exit()
            
            image = screenshot_window(window)

            # Generate random 8 char string for filename
            random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            image_filename = f'{random_string}.png'
            image_path = os.path.join(screenshots_path, image_filename)

            # Save screenshot
            image.save(image_path)
            print(' Captured')

            # No interval specified, exit loop
            if interval is None:
                break
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Quitting program')


if __name__ == '__main__':
    main()
