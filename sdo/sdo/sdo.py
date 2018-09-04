from ortools.linear_solver import pywraplp
import warnings, sys, os, time, copy
from os.path import dirname, join

sys.path.extend([join(dirname(dirname(dirname(__file__))), 'mbp')])
from mbp import OneBagPacker

class ShelfDisplayOptimizer(object):
    """ to find the optimal way to display product on shelves (a global MIO approach)
    Attributes:
        skus_info (dict): skus information (quantity, length)
        m (int): number of shelves to be used
        nl (int): number of layers per shelf can be used
        L (float): length of each layer
        solver (obj): an instance of pywraplp.Solver (our core optimization engine)
    """
    def __init__(self, skus_info, m, nl, L, time_limit=-1):
        self.skus_info = copy.deepcopy(skus_info)
        self.sku_ids = list(skus_info.keys())
        self.n = len(self.sku_ids)  # number of products to be displayed on shelf
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


    def optimize_greedy(self):

        q,l,n,m,nl,L \
            = self.q, self.l, self.n, self.m, self.nl, self.L

        idx_left = list(range(len(q)))

        shelf = 0
        q_left = [q[i] for i in idx_left]

        while len(idx_left) > 0 and shelf < m:

            print(f"Optimizing shelf {shelf}...")

            st = time.time()

            if len([_ for _ in q_left if _ > 1]) > 0:

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

                # approach 2
                w = [q[i] * l[i] for i in idx_left]
                obp = OneBagPacker(weights=w, capacity=L * nl, dg=1000)
                obp.pack()
                s = [idx_left[i] for i in obp.packed_items]  # global index


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
                q_left = [q[i] for i in idx_left]

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
                                self.result[self.inv_idx_dict[idx]]['y'] = [x + self.result[self.inv_idx_dict[idx]]['l']]
                                self.result[self.inv_idx_dict[idx]]['n'] = [self.result[self.inv_idx_dict[idx]]['q']]
                                x += self.result[self.inv_idx_dict[idx]]['l']
                            idx_left = [_ for _ in idx_left if _ not in s]
                            #print(idx_left)


            else:
                # all q left is 1, then simple handle layer by layer
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
                        self.result[self.inv_idx_dict[idx]]['y'] = [x_ + self.result[self.inv_idx_dict[idx]]['l']]
                        self.result[self.inv_idx_dict[idx]]['n'] = [1]
                        x_ += self.result[self.inv_idx_dict[idx]]['l']
                    idx_left = [_ for _ in idx_left if _ not in s]
                    q_left = [q[i] for i in idx_left]
                    if len(idx_left) < 1:
                        break

            # after finishing this shelf
            print(f"Optimizing shelf {shelf}...done ({round(time.time()-st,2)} sec).")
            shelf += 1



        self.num_of_shelf = shelf
        self._output_layout()


        # TODO: analyze the final output


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

        # objective
        solver.Maximize(sum([l_s[i] * n2d[i][k] for i in range(n) for k in range(nl)]) + \
                   .1 * sum([o[i][k] for i in range(n) for k in range(nl)]) + \
                   # sum([10**(-3-k)*l_s[i]*n2d[i][k] for i in range(n) for k in range(nl)]) + \
                   -.01*sum([IX2d[i][k] for i in range(n) for k in range(nl)]) + \
                   -.0001*sum([y[i][k]/L for i in range(n) for k in range(nl)]))


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

    x = x.copy()
    y = y.copy()

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

