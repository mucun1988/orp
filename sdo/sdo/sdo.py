from ortools.linear_solver import pywraplp
import warnings, sys, os, time, copy
from os.path import dirname, join
from PIL import Image, ImageDraw, ImageChops #to plot

sys.path.extend([join(dirname(dirname(dirname(__file__))), 'mbp')])
from mbp import OneBagPacker

class ShelfDisplayOptimizer(object):
    """ to find the optimal way to display product on shelves (a global MIO approach)
    Attributes:
        skus_info (dict): skus information (quantity, length)
        sku_ids (list): sku ids
        n (int): number of diff. products to be displayed on shelf
        idx_dict: dictionary sku_id |-> idx
        inv_idx_dict: dictionary idx |-> sku_id
        q (list): facings
        l (list): lengths
        m (int): number of shelves to be used
        nl (int): number of layers per shelf can be used
        L (float): length of each layer
        result (list): solution
        time_limit: imposed to the solver
        solver (obj): an instance of pywraplp.Solver (our core optimization engine)
    """
    def __init__(self, skus_info, m, nl, L, time_limit=-1):
        self.skus_info = copy.deepcopy(skus_info)
        self.sku_ids = list(skus_info.keys())
        self.n = len(self.sku_ids)
        self.idx_dict = dict(zip(self.sku_ids, range(self.n)))
        self.inv_idx_dict = dict(zip(range(self.n), self.sku_ids))
        self.q = [skus_info[self.inv_idx_dict[i]]['q'] for i in range(self.n)]
        self.l = [skus_info[self.inv_idx_dict[i]]['l'] for i in range(self.n)]
        self.m = m
        self.nl = nl
        self.L = L
        self.result = copy.deepcopy(skus_info)
        self.time_limit=time_limit

        try:
            self.solver = _new_solver('sdo', integer=True)
        except:
            warnings.warn("CPLEX is not found and switch to CBC!")
            self.solver = _new_solver('sdo', integer=True, cplex=False)

        if time_limit > 0:
            self.solver.set_time_limit(time_limit)

    def optimize_global(self):
        """
        solve the problem using global MIO approach
        """
        assert self.m > 0
        q,l,n,m,nl,L, solver \
            = self.q, self.l, self.n, self.m, self.nl, self.L, self.solver

        # define variables
        B1d = [solver.IntVar(0, 1, f'B_{i}') for i in range(m)]
        B2d = [[solver.IntVar(0, 1, f'B_{i}_{j}') for j in range(m)] for i in range(n)]
        B3d = [[[solver.IntVar(0, 1, f'B_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
        n3d = [[[solver.IntVar(0, solver.infinity(), f'n_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]


        Left = [[[[solver.IntVar(0, 1, f'L_{i}_{ip}_{j}_{k}') for k in range(nl)] for j in range(m)] \
                                                                   for ip in range(n)] for i in range(n)]

        x = [[[solver.NumVar(0.0, L, f'x_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
        y = [[[solver.NumVar(0.0, L, f'y_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
        h = [[[solver.NumVar(0.0, L, f'h_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
        t = [[[solver.NumVar(0.0, L, f't_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
        o = [[[solver.NumVar(0.0, L, f'o_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]

        # must put everything on shelf
        for i in range(n):
            solver.Add(sum([n3d[i][j][k] for j in range(m) for k in range(nl)]) == q[i])

        # i, ip: whose on the left side
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for k in range(nl):
                        if i != ip:
                            solver.Add(
                                Left[i][ip][j][k] + Left[ip][i][j][k] + (1 - B3d[i][j][k]) + (1 - B3d[ip][j][k]) >= 1)

        # no collision
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for k in range(nl):
                        solver.Add(y[i][j][k] + Left[i][ip][j][k] * L <= x[ip][j][k] + L)

        # must put one of the shelf
        for i in range(n):
            solver.Add(sum([B2d[i][j] for j in range(m)]) == 1)

        # connected
        for i in range(n):
            for j in range(m):
                for k1 in range(nl):
                    for k2 in range(nl):
                        for k3 in range(nl):
                            if k1 < k2 and k2 < k3:
                                solver.Add(B3d[i][j][k1] - B3d[i][j][k2] + B3d[i][j][k3] <= 1)

        # overlapping area
        for i in range(n):
            for j in range(m):
                for k in range(nl):
                    for kp in range(nl):
                        solver.Add(t[i][j][k] - (1 - B3d[i][j][k]) * L <= y[i][j][kp] + (1 - B3d[i][j][kp]) * L)

        for i in range(n):
            for j in range(m):
                for k in range(nl):
                    for kp in range(nl):
                        solver.Add(h[i][j][k] + (1 - B3d[i][j][k]) * L >= x[i][j][kp] - (1 - B3d[i][j][kp]) * L)

        for i in range(n):
            for j in range(m):
                for k in range(nl):
                    solver.Add(y[i][j][k] - x[i][j][k] == l[i] * n3d[i][j][k])
                    solver.Add(o[i][j][k] == t[i][j][k] - h[i][j][k]) # overlapping area
                    solver.Add(o[i][j][k] >= l[i] * B3d[i][j][k])     # must be connected
                    solver.Add(y[i][j][k] <= L * B3d[i][j][k])
                    solver.Add(t[i][j][k] <= y[i][j][k])
                    solver.Add(h[i][j][k] <= t[i][j][k])
                    solver.Add(x[i][j][k] <= h[i][j][k])
                    solver.Add(B3d[i][j][k] <= B2d[i][j])

        for i in range(n):
            for j in range(m):
                solver.Add(B2d[i][j] <= B1d[j])

        # 1. minimize the number of shelves
        # 2. maximize the overlapping area
        solver.Maximize(-sum([B1d[j] for j in range(m)]) + \
                        0.0001 * sum([o[i][j][k] for i in range(n) for j in range(m) for k in range(nl)]))

        result_status=solver.Solve()


        self.optimal= (result_status == pywraplp.Solver.OPTIMAL)
        self.x = _sol_val(x)
        self.y = _sol_val(y)
        self.n3d = _sol_val(n3d)
        self.B1d = _sol_val(B1d)
        self.B2d = _sol_val(B2d)
        self.B3d = _sol_val(B3d)
        self.num_of_shelf = m
        self._post_process_global()
        self._output_layout()


    def _post_process_global(self):
        for i in range(self.n):
            shelf_layer = [(j, k) for j in range(self.m) for k in range(self.nl) if self.B3d[i][j][k] == 1]
            self.result[self.inv_idx_dict[i]]['shelf'] = [item[0] for item in shelf_layer]
            self.result[self.inv_idx_dict[i]]['layer'] = [item[1] for item in shelf_layer]
            self.result[self.inv_idx_dict[i]]['x'] = [self.x[i][j][k] for (j, k) in shelf_layer]
            self.result[self.inv_idx_dict[i]]['y'] = [self.y[i][j][k] for (j, k) in shelf_layer]
            self.result[self.inv_idx_dict[i]]['n'] = [self.n3d[i][j][k] for (j, k) in shelf_layer]

    def _output_layout(self):
        results = self.result
        display_result = [[{'skus': [], 'x': [], 'y': [], 'n': []} for j in range(self.nl)]
                                                                        for i in range(self.num_of_shelf)]

        for sku_id in results.keys():
            result = results[sku_id]
            if 'shelf' in result.keys(): # skus allocated
                shelf = result['shelf']
                layer = result['layer']
                x = result['x']
                y = result['y']
                n = result['n']
                for i in range(len(result['shelf'])):
                    display_result[shelf[i]][layer[i]]['skus'].append(sku_id)
                    display_result[shelf[i]][layer[i]]['x'].append(x[i])
                    display_result[shelf[i]][layer[i]]['y'].append(y[i])
                    display_result[shelf[i]][layer[i]]['n'].append(n[i])

        # sorting
        for i in range(self.num_of_shelf):
            for j in range(self.nl):
                s = display_result[i][j]['x']
                if len(s) > 1:
                    idx = sorted(range(len(s)), key=lambda k: s[k])
                    display_result[i][j]['skus'] = list(map(lambda _: display_result[i][j]['skus'][_], idx))
                    display_result[i][j]['x'] = list(map(lambda _: display_result[i][j]['x'][_], idx))
                    display_result[i][j]['y'] = list(map(lambda _: display_result[i][j]['y'][_], idx))
                    display_result[i][j]['n'] = list(map(lambda _: display_result[i][j]['n'][_], idx))

        shelf_utilization = [[[] for j in range(self.nl)] for i in range(self.num_of_shelf)]

        for i in range(self.num_of_shelf):
            for j in range(self.nl):
                x = display_result[i][j]['x']
                y = display_result[i][j]['y']
                if len(x) == 0:
                    shelf_utilization[i][j] = None
                else:
                    shelf_utilization[i][j] = sum([y[i]-x[i] for i in range(len(x))])/self.L

        self.layout = display_result
        self.shelf_utilization = shelf_utilization

    def plot_shelf_display(self, shelf, image_folder='/Users/matthew.mu/dat/images', save_file=None):

        assert shelf < self.num_of_shelf

        im = _plot_result(self.layout[shelf], image_folder=image_folder)

        if save_file == None:
            im.save('./sdo/fig/shelf_'+str(shelf)+'.jpg', 'JPEG')
        else:
            im.save(save_file, 'JPEG')

        return im

    def plot_layout_into_one(self, num_row):
        return _plot_all_results(self.layout, num_row=num_row)

    def optimize_greedy(self, threshold=1):
        """
        solve the problem shelf by shelf using MIO
        """

        q,l,n,m,nl,L \
            = self.q, self.l, self.n, self.m, self.nl, self.L

        idx_left = list(range(len(q)))
        # q_left = [q[i] for i in idx_left]
        idx_left_hard = [i for i in idx_left if q[i] >= threshold or q[i] * l[i] > L]

        shelf = 0

        while len(idx_left) > 0 and shelf < m:

            print(f"Optimizing shelf {shelf}...")

            st = time.time()

            # if len([_ for _ in  if _ > threshold]) > 0: # check quantity>=2 left or not
            if len(idx_left_hard) > 0:

                # approach 1
                # # select s to be displayed
                # w = [q[i] * l[i] for i in idx_left]
                # obp = OneBagPacker(weights=w, capacity=L * (nl+1), dg=1000)
                # obp.pack()
                # s1 = [idx_left[i] for i in obp.packed_items]  # global index
                #
                # obp = OneBagPackerOpp(weights=w, capacity=L * 2, dg=1000)
                # obp.pack()
                # s2 = [idx_left[i] for i in obp.packed_items]
                #
                # s1.extend(s2)
                #
                # s = list(set(s1))
                # select s to be displayed

                # approach 2 (1 can actually change to anything)
                # idx_left_2 = [i for i in idx_left if q[i] >= threshold or q[i]*l[i] > L]
                assert len(idx_left_hard) > 0
                w = [q[i] * l[i] for i in idx_left_hard]
                obp = OneBagPacker(weights=w, capacity=L * nl, dg=1000)
                obp.pack()
                s = [idx_left_hard[i] for i in obp.packed_items]  # global index

                # optimize display using products in s
                q_s = [q[i] for i in s]
                l_s = [l[i] for i in s]

                osdo = OneShelfDisplayOptimizer(q_s, l_s, nl, L, time_limit=self.time_limit)

                osdo.optimize()

                assert osdo.feasible or osdo.optimal

                if osdo.feasible:
                    warnings.warn("Feasible but perhaps suboptimal!")


                idx_put = [s[i] for i in range(len(s)) if osdo.B1d[i] > 0] # global index
                ids_put = [self.inv_idx_dict[i] for i in idx_put]

                x_layer = [[] for i in range(nl)]
                y_layer = [[] for i in range(nl)]

                for i in range(len(s)): # i is local
                    if osdo.B1d[i] > 0:
                        shelf_list = []
                        layer_list = []
                        x_list = []
                        y_list = []
                        n_list = []
                        for k in range(nl):
                            if osdo.B2d[i][k] > 0:
                                if osdo.n2d[i][k] > 0:
                                    shelf_list.append(shelf)
                                    layer_list.append(k)
                                    x_list.append(osdo.x[i][k])
                                    y_list.append(osdo.y[i][k])
                                    n_list.append(osdo.n2d[i][k])
                                    x_layer[k].append(osdo.x[i][k])
                                    y_layer[k].append(osdo.y[i][k])

                        self.result[self.inv_idx_dict[s[i]]]['shelf'] = shelf_list
                        self.result[self.inv_idx_dict[s[i]]]['layer'] = layer_list
                        self.result[self.inv_idx_dict[s[i]]]['x'] = x_list
                        self.result[self.inv_idx_dict[s[i]]]['y'] = y_list
                        self.result[self.inv_idx_dict[s[i]]]['n'] = n_list

                del osdo
                idx_left = [x for x in idx_left if x not in idx_put]

                # put in small stuff
                for i in range(nl):
                    if len(idx_left) > 0:
                        empty_space_list = _find_space(x_layer[i], y_layer[i], self.L)
                        for j in range(len(empty_space_list)):
                            x,y = empty_space_list[j]
                            w = [q[i] * l[i] for i in idx_left]
                            obp = OneBagPacker(weights=w, capacity=y-x, dg=1000)
                            obp.pack()
                            s = [idx_left[i] for i in obp.packed_items]  # global index
                            #print(s)
                            for idx in s:
                                self.result[self.inv_idx_dict[idx]]['shelf'] = [shelf]
                                self.result[self.inv_idx_dict[idx]]['layer'] = [i]
                                self.result[self.inv_idx_dict[idx]]['x'] = [x]
                                self.result[self.inv_idx_dict[idx]]['y'] = [x + \
                                    self.result[self.inv_idx_dict[idx]]['l']*self.result[self.inv_idx_dict[idx]]['q']]
                                self.result[self.inv_idx_dict[idx]]['n'] = [self.result[self.inv_idx_dict[idx]]['q']]
                                x += self.result[self.inv_idx_dict[idx]]['l']*self.result[self.inv_idx_dict[idx]]['q']
                            idx_left = [_ for _ in idx_left if _ not in s]
                            #print(idx_left)


            else:
                print("    easy case.")
                # all q left is 1 (could be more flexible), then simple handle layer by layer
                for layer in range(nl):
                    w = [q[i] * l[i] for i in idx_left]
                    obp = OneBagPacker(weights=w, capacity=L, dg=1000)
                    obp.pack()
                    s = [idx_left[i] for i in obp.packed_items]  # global index
                    x_ = 0
                    for idx in s:
                        self.result[self.inv_idx_dict[idx]]['shelf'] = [shelf]
                        self.result[self.inv_idx_dict[idx]]['layer'] = [layer]
                        self.result[self.inv_idx_dict[idx]]['x'] = [x_]
                        self.result[self.inv_idx_dict[idx]]['y'] = [x_ + \
                            self.result[self.inv_idx_dict[idx]]['l'] * self.result[self.inv_idx_dict[idx]]['q']]
                        self.result[self.inv_idx_dict[idx]]['n'] = [self.result[self.inv_idx_dict[idx]]['q']]
                        x_ += self.result[self.inv_idx_dict[idx]]['l'] * self.result[self.inv_idx_dict[idx]]['q']
                    idx_left = [_ for _ in idx_left if _ not in s]

                    if len(idx_left) < 1:
                        break

            # after finishing this shelf
            print(f"Optimizing shelf {shelf}...done ({round(time.time()-st,2)} sec).")
            shelf += 1
            # q_left = [q[i] for i in idx_left]
            idx_left_hard = [i for i in idx_left if q[i] >= threshold or q[i] * l[i] > L]

        self.num_of_shelf = shelf
        self._output_layout()


class OneShelfDisplayOptimizer(object):
    """ to find the optimal way to display product to shelves
    Attributes:
        q (list): product quantities
        l (list): product lengths
        nl (int): number of layers per shelf can be used
        L (float): length of each layer
        solver (obj): an instance of pywraplp.Solver (our core optimization engine)
    """
    def __init__(self, q, l, nl, L, time_limit=-1):
        assert len(q)==len(l), 'len(q)!=len(l)'
        self.q = q
        self.l = l
        self.n = len(q) # number of products to be displayed on shelf
        self.nl = nl
        self.L = L
        self.time_limit=time_limit
        try:
            self.solver = _new_solver('sdo-g', integer=True)
        except:
            warnings.warn("CPLEX is not found and switch to CBC!")
            self.solver = _new_solver('sdo-g', integer=True, cplex=False)

        if time_limit > 0:
            self.solver.set_time_limit(time_limit)

    def optimize(self):

        q_s,l_s,n,nl,L,solver \
            = self.q, self.l, self.n, self.nl, self.L, self.solver

        # define variables
        B1d = [solver.IntVar(0, 1, f'B_{i}') for i in range(n)]
        B2d = [[solver.IntVar(0, 1, f'B_{i}_{j}') for j in range(nl)] for i in range(n)]
        IX2d = [[solver.IntVar(0, 1, f'IX_{i}_{k}') for k in range(nl)] for i in range(n)]
        Left = [[[solver.IntVar(0, 1, f'L_{i}_{ip}_{k}') for k in range(nl)] for ip in range(n)] for i in range(n)]

        n2d = [[solver.IntVar(0, q_s[i], f'n_{i}_{k}') for k in range(nl)] for i in range(n)]

        x = [[solver.NumVar(0.0, L, f'x_{i}_{k}') for k in range(nl)] for i in range(n)]
        y = [[solver.NumVar(0.0, L, f'y_{i}_{k}') for k in range(nl)] for i in range(n)]
        h = [[solver.NumVar(0.0, L, f'h_{i}_{k}') for k in range(nl)] for i in range(n)]
        t = [[solver.NumVar(0.0, L, f't_{i}_{k}') for k in range(nl)] for i in range(n)]
        o = [[solver.NumVar(0.0, L, f'o_{i}_{k}') for k in range(nl)] for i in range(n)]

        # constraints

        # quantity
        for i in range(n):
            solver.Add(sum([n2d[i][k] for k in range(nl)]) == q_s[i] * B1d[i])

        # no collisions
        for i in range(n):
            for ip in range(n):
                for k in range(nl):
                    if i != ip:
                        solver.Add(Left[i][ip][k] + Left[ip][i][k] + (1 - B2d[i][k]) + (1 - B2d[ip][k]) >= 1)

        for i in range(n):
            for ip in range(n):
                for k in range(nl):
                    solver.Add(y[i][k] + Left[i][ip][k] * L <= x[ip][k] + L)

        # connected
        for i in range(n):
            for k1 in range(nl):
                for k2 in range(nl):
                    for k3 in range(nl):
                        if k1 < k2 and k2 < k3:
                            solver.Add(B2d[i][k1] - B2d[i][k2] + B2d[i][k3] <= 1)

        # overlapping length (o)
        for i in range(n):
            for k in range(nl):
                for kp in range(nl):
                    solver.Add(t[i][k] - (1 - B2d[i][k]) * L <= y[i][kp] + (1 - B2d[i][kp]) * L)

        for i in range(n):
            for k in range(nl):
                for kp in range(nl):
                    solver.Add(h[i][k] + (1 - B2d[i][k]) * L >= x[i][kp] - (1 - B2d[i][kp]) * L)

        for i in range(n):
            for k in range(nl):
                solver.Add(y[i][k] - x[i][k] == l_s[i] * n2d[i][k])
                solver.Add(o[i][k] == t[i][k] - h[i][k])
                # solver.Add(o[i][k] >= l_s[i] * B2d[i][k]) # must be connected
                solver.Add(y[i][k] <= L * B2d[i][k])
                solver.Add(y[i][k] >= t[i][k])
                solver.Add(t[i][k] >= h[i][k])
                solver.Add(h[i][k] >= x[i][k])
                solver.Add(x[i][k] <= L * IX2d[i][k])
                solver.Add(B2d[i][k] <= B1d[i])
                solver.Add(B2d[i][k] <= n2d[i][k]) # enforce consistent

        # objective
        solver.Maximize(sum([l_s[i] * n2d[i][k] for i in range(n) for k in range(nl)]) + \
                   .1 * sum([o[i][k] for i in range(n) for k in range(nl)]) + \
                   -.01*sum([IX2d[i][k] for i in range(n) for k in range(nl)]) + \
                   -.0001*sum([y[i][k]/L for i in range(n) for k in range(nl)]) + \
                   sum([10 ** (- k) * l_s[i] * n2d[i][k] / L for i in range(n) for k in range(nl)]))


        result_status = solver.Solve()

        self.optimal = (result_status == pywraplp.Solver.OPTIMAL)
        self.feasible = (result_status == pywraplp.Solver.FEASIBLE)
        self.x = _sol_val(x)
        self.y = _sol_val(y)
        self.n2d = _sol_val(n2d)
        self.B1d = _sol_val(B1d)
        self.B2d = _sol_val(B2d)

#
# utility functions
def _new_solver(name, integer=False, cplex=True):
  return pywraplp.Solver(name,
    pywraplp.Solver.CPLEX_MIXED_INTEGER_PROGRAMMING
      if integer and cplex else pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
      if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

def _sol_val(x):
  if type(x) is not list:
    return 0 if x is None \
             else x if isinstance(x,(int,float)) \
                    else x.SolutionValue() if x.Integer() is False \
                                           else int(x.SolutionValue())
  elif type(x) is list:
    return [_sol_val(e) for e in x]

def _obj_val(x):
  return x.Objective().Value()

def _find_space(x,y, L, nd=2):

    assert len(x) == len(y)

    if len(x) == 0:
        return [[0, L]]

    x = copy.deepcopy(x)
    y = copy.deepcopy(y)

    # sorted first
    idx = sorted(range(len(x)), key=lambda k: x[k])
    x = list(map(lambda _: x[_], idx))
    y = list(map(lambda _: y[_], idx))

    x = x + [L]
    y = [0] + y

    result = []

    for i in range(len(x)):
        if round(y[i], nd) < round(x[i], nd):
            result.append([y[i], x[i]])

    return result

def _trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def _plot_result(result, shelf_length=48, shelf_height=10, scale=10, image_folder='/Users/matthew.mu/dat/images'):
    # all units in inch

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
            sku_length = (y[i] - x[i]) / n[i]
            try:
                pil_im = Image.open(os.path.join(image_folder, skus[i] + '.jpg'))
            except:
                pil_im = Image.new('RGB', (100, 200), (211, 211, 211))
                dr = ImageDraw.Draw(pil_im)
                dr.rectangle(((10, 10), (90, 190)), outline=sku_color_dict[skus[i]])
                dr.text((15, 15), f"{skus[i]}", fill="black")

            prod_im = _trim(pil_im).resize((round(sku_length * scale), shelf_height*scale), Image.ANTIALIAS)
            x_ = x[i]
            for j in range(n[i]):  # n[i] times
                im.paste(prod_im, (round(x_ * scale), shelf_height*scale * layer))
                x_ += sku_length

    return im

def _plot_all_results(layout, num_row=1):

    images = list()
    for i in range(len(layout)):
        images.append(_plot_result(layout[i]))

    widths, heights = zip(*(i.size for i in images))

    num_col = int(np.ceil(len(images) / num_row))

    max_height = max(heights)
    max_width = max(widths)

    total_width = max_width * num_col + 10 * (num_col - 1)
    total_height = max_height * num_row + 10 * (num_row - 1)

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

    return new_im

def _trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

#TODO: need some preprocess functions
def create_skus_info(skus, q, l):

    assert len(skus) == len(q) == len(q)
    return dict({str(skus[i]):{'q': q[i],'l': l[i]} for i in range(len(skus))})

