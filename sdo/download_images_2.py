from PIL import Image, ImageDraw, ImageChops
import os, sys
from io import BytesIO
from os import path

import pandas, requests

image_folder = '/Users/matthew.mu/dat/images'

image_url_file = '/Users/matthew.mu/dat/roger/sku_data.csv'

df = pandas.read_csv(image_url_file)

OLD_NBRs = list(df['old_nbr'].values)
UPCs = list(df['upc_nbr'].values)

def _trim(im):
    """
    remove white borders
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

for i in range(len(OLD_NBRs)):
    OLD_NBR = OLD_NBRs[i]
    if OLD_NBR is not None:
        image_url = 'http://image-server.item.prodcd5.walmart.com:8080/img?upc=' + str(int(UPCs[i]))
        file_to_save = os.path.join(image_folder, str(OLD_NBR)+'.jpg')
        if path.exists(file_to_save):
            print(f'{i}: exist already!')
        else:
            try:
                response = requests.get(image_url)
                img = _trim(Image.open(BytesIO(response.content)))
                img.save( 'JPEG')
                print(f'{i}: successful!')
            except:
                print(f'{i}: failed!')
                pass


# image_url = 'http://image-server.item.prodcd5.walmart.com:8080/img?upc=88491212611'
# response = requests.get(image_url)
# img = Image.open(BytesIO(response.content))