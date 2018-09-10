import sys
from PIL import Image
from sdo.sdo import _plot_result
import pickle

with open("/Users/matthew.mu/dat/layout.txt", "rb") as fp:   # Unpickling
    layout = pickle.load(fp)

for i in range(len(layout)):
    _plot_result(layout[i]).show()

images = list()
for i in range(len(layout)):
    images.append(_plot_result(layout[i]))

widths, heights = zip(*(i.size for i in images))

num_row = 1
num_col = int(np.ceil(len(images)/num_row))

max_height = max(heights)
max_width = max(widths)

total_width = max_width*num_col + 10*(num_col-1)
total_height = max_height*num_row + 10*(num_row-1)

new_im = Image.new('RGB', (total_width, total_height))



x_offset = 0
y_offset = 0

cnt = 0
for im in images:

    new_im.paste(im, (x_offset, y_offset))
    x_offset += max_width + 10
    cnt += 1
    if cnt >= num_col:
        cnt = 0
        x_offset = 0
        y_offset += max_height + 10

new_im.show()


with open("/Users/matthew.mu/dat/layout_global.txt", "rb") as fp:   # Unpickling
    layout = pickle.load(fp)

_plot_result(layout[0]).show()