import numpy as np
from ortools.linear_solver import pywraplp
import warnings, copy, os
from PIL import Image, ImageDraw, ImageChops #to plot

def new_solver(name, integer=False, cplex=True):
  return pywraplp.Solver(name,
    pywraplp.Solver.CPLEX_MIXED_INTEGER_PROGRAMMING
      if integer and cplex else pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
      if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

def sol_val(x):
  if type(x) is not list:
    return 0 if x is None \
             else x if isinstance(x,(int,float)) \
                    else x.SolutionValue() if x.Integer() is False \
                                           else int(x.SolutionValue())
  elif type(x) is list:
    return [sol_val(e) for e in x]

def obj_val(x):
  return x.Objective().Value()

def create_layout_from_result(result, nl):
    layout = [[{'skus': [], 'x': [], 'y': [], 'n': []} for j in range(nl)]
                      for i in range(1)]

    for sku_id in result.keys():
        rs = result[sku_id]
        if 'shelf' in rs.keys():  # skus allocated
            shelf = rs['shelf']
            layer = rs['layer']
            x = rs['x']
            y = rs['y']
            n = rs['n']
            for i in range(len(rs['shelf'])):
                layout[shelf[i]][layer[i]]['skus'].append(sku_id)
                layout[shelf[i]][layer[i]]['x'].append(x[i])
                layout[shelf[i]][layer[i]]['y'].append(y[i])
                layout[shelf[i]][layer[i]]['n'].append(n[i])

    # sorting
    for i in range(1):
        for j in range(nl):
            s = layout[i][j]['x']
            if len(s) > 1:
                idx = sorted(range(len(s)), key=lambda k: s[k])
                layout[i][j]['skus'] = list(map(lambda _: layout[i][j]['skus'][_], idx))
                layout[i][j]['x'] = list(map(lambda _: layout[i][j]['x'][_], idx))
                layout[i][j]['y'] = list(map(lambda _: layout[i][j]['y'][_], idx))
                layout[i][j]['n'] = list(map(lambda _: layout[i][j]['n'][_], idx))

    return layout

def plot_layout(layout, shelf_length=48, shelf_height=10, scale=10, image_folder='/Users/matthew.mu/dat/images'):
    """
    plot the layout of one shelf (all units in inch)
    """
    result = layout

    nl = len(result)

    # plot shelf
    im = Image.new('RGB', [shelf_length*scale, nl*shelf_height*scale], (211, 211, 211))
    draw = ImageDraw.Draw(im)

    for i in range(nl):
        draw.line((0, shelf_height*scale * i + shelf_height*scale, shelf_length*scale, shelf_height*scale * i + shelf_height*scale),
                  fill=(0, 0, 0), width=2)

    # result to display
    colors = ['blue', 'green', 'red', 'yellow', 'black', 'brown']

    sku_list = list(set([item for sl in [result[i]['skus'] for i in range(len(result))] for item in sl]))
    sku_color_dict = dict({sku_list[i]: colors[i % len(colors)] for i in range(len(sku_list))})

    for layer in range(len(result)):

        result_layer = result[layer]
        n = result_layer['n']
        skus = result_layer['skus']
        x = result_layer['x']
        y = result_layer['y']
        for i in range(len(n)):
            if n[i] > 0 and y[i] - x[i] > 0:
                sku_length = (y[i] - x[i]) / n[i]
                try:
                    pil_im = Image.open(os.path.join(image_folder, str(skus[i]) + '.jpg'))
                except:
                    pil_im = Image.new('RGB', (100, 200), (211, 211, 211))
                    dr = ImageDraw.Draw(pil_im)
                    dr.rectangle(((10, 10), (90, 190)), outline=sku_color_dict[skus[i]])
                    # dr.text((15, 15), f"{skus[i]}", fill="black")

                prod_im = _trim(pil_im).resize((round(sku_length * scale), shelf_height*scale), Image.ANTIALIAS)
                dr = ImageDraw.Draw(prod_im)
                dr.text((2.5, 2.5), f"{skus[i]}", fill="black")
                x_ = x[i]
                for j in range(n[i]):  # n[i] times
                    im.paste(prod_im, (round(x_ * scale), shelf_height*scale * layer))
                    x_ += sku_length

    return im

# plotting functions
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




