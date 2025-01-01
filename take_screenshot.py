import argparse
import os
import random
import time
import screen


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
            # Take screenshot of window
            image = screen.take_screenshot(window_name)
            
            # Generate random 8 char string for filename
            random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            image_filename = f'{random_string}.png'
            image_path = os.path.join(screenshots_path, image_filename)

            # Save screenshot
            image.save(image_path)

            # No interval specified, exit loop
            if interval is None:
                break
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Quitting program')


if __name__ == '__main__':
    main()
