# Removes the random 8 characters that get added to the
# filename when it is exported from Label Studio since
# it adds it on to the front each time it's exported

import os

image_dir = os.path.join('data', 'images')
if not os.path.exists(image_dir):
    print('Specified directory does not exist')
    exit(0)

for filename in os.listdir(image_dir):
    pass