from PIL import Image, ImageDraw, ImageChops

import os, sys
image_folder = '/Users/matthew.mu/dat/images'
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# plot shelf
im = Image.new('RGB', [480, 500], (211, 211, 211))
draw = ImageDraw.Draw(im)
nl = 5
for i in range(nl):
    draw.line((0,100*i+100, 480, 100*i+100), fill=(0,0,0), width=2)
im.show()

# result = [{'n': [4, 1, 1],
#   'skus': ['567890511', '552582147', '564690585'],
#   'x': [0.0, 37.6, 41.660000000000004],
#   'y': [37.6, 41.660000000000004, 48.0]},
#  {'n': [4, 1, 1],
#   'skus': ['567890511', '566823238', '552069744'],
#   'x': [0.0, 37.6, 43.910000000000004],
#   'y': [37.6, 43.910000000000004, 47.970000000000006]},
#  {'n': [2], 'skus': ['567900028'], 'x': [0.0], 'y': [47.00000000000001]},
#  {'n': [2], 'skus': ['567900028'], 'x': [0.0], 'y': [47.00000000000001]},
#  {'n': [2], 'skus': ['567900028'], 'x': [0.0], 'y': [47.00000000000001]}]


result = [{'n': [6], 'skus': ['565546425'], 'x': [0.0], 'y': [46.62]},
 {'n': [6], 'skus': ['565546425'], 'x': [0.0], 'y': [46.62]},
 {'n': [6], 'skus': ['565546425'], 'x': [0.0], 'y': [46.62]},
 {'n': [2, 3],
  'skus': ['555517895', '568919318'],
  'x': [0.0, 18.0],
  'y': [18.0, 46.5]},
 {'n': [1, 3],
  'skus': ['555517895', '568919318'],
  'x': [0.0, 17.999999999999975],
  'y': [9.0, 46.49999999999997]}]

# result = [{'n': [1, 1, 3, 1],
#   'skus': ['9257630', '556629971', '567890507', '552582148'],
#   'x': [0.0, 7.769999999999999, 15.539999999999973, 43.73999999999998],
#   'y': [7.769999999999999,
#    15.532999999999998,
#    43.73999999999998,
#    47.79999999999998]},
#  {'n': [2, 3, 1],
#   'skus': ['9257630', '567890507', '552582146'],
#   'x': [0.0, 15.539999999999997, 43.74],
#   'y': [15.539999999999997, 43.74, 47.800000000000004]},
#  {'n': [1, 4],
#   'skus': ['9203185', '567890518'],
#   'x': [0.0, 9.562],
#   'y': [9.562, 47.162000000000006]},
#  {'n': [1, 4],
#   'skus': ['9203185', '567890518'],
#   'x': [0.0, 9.562],
#   'y': [9.562, 47.162000000000006]},
#  {'n': [1, 4],
#   'skus': ['9203185', '567890518'],
#   'x': [0.0, 9.562],
#   'y': [9.562, 47.162000000000006]}]




# result to display
colors = ['blue', 'green', 'red', 'yellow', 'black', 'brown']
color_idx = 0
sku_list = list(set([item for sl in [result[i]['skus'] for i in range(len(result))] for item in sl]))
sku_color_dict = dict({sku_list[i]: colors[i%len(colors)] for i in range(len(sku_list))})

for layer in range(len(result)):

    result_layer = result[layer]
    n = result_layer['n']
    skus = result_layer['skus']
    x = result_layer['x']
    y = result_layer['y']
    for i in range(len(n)):
        print(skus[i])
        sku_length = (y[i]-x[i])/n[i]
        try:
            pil_im = Image.open(os.path.join(image_folder, skus[i] + '.jpg'))
        except:
            pil_im = Image.new('RGB', (100,200), (211, 211, 211))
            dr = ImageDraw.Draw(pil_im)
            dr.rectangle(((10, 10), (90, 190)), outline=sku_color_dict[skus[i]])
            dr.text((15, 15), f"{skus[i]}", fill="black")
            color_idx += 1

        prod_im = trim(pil_im).resize((round(sku_length*10), 100), Image.ANTIALIAS)
        x_ = x[i]
        print(x_)
        for j in range(n[i]): # n[i] times
            im.paste(prod_im, (round(x_*10), 100*layer))
            x_ += sku_length

im.show()

im.save('./sdo/fig/shelf_3.jpg' ,'JPEG')





