try:
    # Attempt relative import (works when run as part of a package)
    from . import config
except ImportError:
    # Fallback to absolute import (works when run independently)
    import config

from PIL import Image
import Quartz.CoreGraphics as CG


#################### FIND WINDOW BY NAME ####################
def find_window_by_name(window_name):
    # Only gets windows if script is run in VSCode (doesn't work in Terminal for some reason)
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


#################### TAKE SCREENSHOT ####################
def take_screenshot(window_name: str) -> Image:
    print('Taking screenshot...', end='', flush=True)
    window = find_window_by_name(window_name)
    if not window:
        print('No window found with the name:', window_name)
        exit()
    
    image = screenshot_window(window)
    print(' Captured')

    return image
