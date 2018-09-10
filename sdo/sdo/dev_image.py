from PIL import Image, ImageDraw, ImageChops
import os, sys

image_folder = '/Users/matthew.mu/dat/images'

pil_im = Image.open(os.path.join(image_folder, '567890511'+'.jpg'))
pil_im.thumbnail((128,128))
pil_im.show()

pil_im = Image.open(os.path.join(image_folder, '567890511'+'.jpg'))
out = pil_im.resize((128,128))
out.show()

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

pil_im = Image.open(os.path.join(image_folder, '567890511'+'.jpg'))
out = trim(pil_im)
out.show()

for infile in os.listdir(image_folder):
    outfile = os.path.splitext(infile)[0] + '.jpg'
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print(f'cannot convert {infile}.')

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

imlist = get_imlist(image_folder)

for infile in imlist:
    print(infile)
    trim(Image.open(infile)).save(infile, 'JPEG')




result = [{'n': [4, 1, 1],
  'skus': ['567890511', '552582147', '564690585'],
  'x': [0.0, 37.6, 41.660000000000004],
  'y': [37.6, 41.660000000000004, 48.0]},
 {'n': [4, 1, 1],
  'skus': ['567890511', '566823238', '552069744'],
  'x': [0.0, 37.6, 43.910000000000004],
  'y': [37.6, 43.910000000000004, 47.970000000000006]},
 {'n': [2], 'skus': ['567900028'], 'x': [0.0], 'y': [47.00000000000001]},
 {'n': [2], 'skus': ['567900028'], 'x': [0.0], 'y': [47.00000000000001]},
 {'n': [2], 'skus': ['567900028'], 'x': [0.0], 'y': [47.00000000000001]}]

)




from PIL import Image, ImageDraw, ImageChops

import os, sys

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

im = Image.new('RGB', [480, 500], (211, 211, 211))
draw = ImageDraw.Draw(im)
nl = 5
for i in range(nl):
    draw.line((0,100*i+100, 480, 100*i+100), fill=(0,0,0), width=2)
im.show()

image_folder = '/Users/matthew.mu/dat/images'
pil_im = Image.open(os.path.join(image_folder, '567890511'+'.jpg'))
prod_im = trim(pil_im).resize((round(37.6/4*10), 100), Image.ANTIALIAS)


im.paste(prod_im, (0,0))
im.paste(prod_im, (round(37.6/4*10),0))
im.paste(prod_im, (round(37.6/4*2*10),0))
im.paste(prod_im, (round(37.6/4*3*10),0))

im.show()

image_folder = '/Users/matthew.mu/dat/images'
pil_im = Image.open(os.path.join(image_folder, '552582147'+'.jpg'))
prod_im = trim(pil_im).resize((round(5*10), 100), Image.ANTIALIAS)
im.paste(prod_im, (round(37.6*10),0))
im.show()

image_folder = '/Users/matthew.mu/dat/images'
pil_im = Image.open(os.path.join(image_folder, '564690585'+'.jpg'))
prod_im = trim(pil_im).resize((round(6.34*10), 100), Image.ANTIALIAS)
im.paste(prod_im, (round(41.66*10),0))
im.show()


from PIL import Image, ImageFont, ImageDraw

im = Image.new('RGB', (100,200), (211, 211, 211))
dr = ImageDraw.Draw(im)
dr.rectangle(((10,10),(90,190)), outline = "blue")
dr.text((40,95), "test", fill="black")

trim(im).show()
