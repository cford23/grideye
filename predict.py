try:
    # Attempt relative import (works when run as part of a package)
    from . import config
except ImportError:
    # Fallback to absolute import (works when run independently)
    import config

import os
from PIL import Image
import utils


image_filename = 'f36e6c12-5ytnuhwh.png'
image_path = os.path.join(config.DATA_DIR, 'images', image_filename)
image = Image.open(image_path)

# Display image with correct boxes
utils.display_original_image(image_path, title='Image with correct boxes')

# Get object predictions and organize into correct format
model = utils.load_model()
predictions = utils.detect_objects(model, image)
predictions = utils.organize_predictions(predictions)

# Display image with predicted boxes
utils.display_image(image_path, predictions, title='Image with predicted boxes')

# Display image with filtered prediction boxes
filtered_preds = utils.filter_predictions(predictions)
utils.display_image(image_path, filtered_preds, title='Image with filtered prediction boxes')
