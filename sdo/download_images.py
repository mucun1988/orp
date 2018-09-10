from PIL import Image, ImageDraw, ImageChops
import os, sys
from io import BytesIO

import pandas, requests

image_folder = '/Users/matthew.mu/dat/images'

image_url_file = '/Users/matthew.mu/dat/image_urls.csv'

df = pandas.read_csv(image_url_file)

skus = list(df['vendorsku'].values)
image_urls = list(df['thorImageUrl'].values)

for i in range(len(skus)):
    sku = skus[i]
    if sku is not None:
        image_url = image_urls[i]
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(image_folder, 'test', str(sku)+'.jpg'), 'JPEG')
        except:
            pass