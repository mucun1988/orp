import numpy as np
import warnings, copy, os
from oso.utils import *

# preprocess
# test json of skus info
skus_info = {
    '568919318': {'l': 9.5, 'q': 20, 'brand': 'Cheerios'},
    '568306983': {'l': 9.4, 'q': 5, 'brand': 'Cheerios'},
    '552581066': {'l': 7.65, 'q': 10, 'brand': 'Cheerios'},
    '569728847': {'l': 9.5, 'q': 3, 'brand': 'Kix'},
    '569728848': {'l': 9.5, 'q': 3, 'brand': 'Kix'},
    '567900044': {'l': 7.77, 'q': 3, 'brand': 'Kellogg'},
    '568050117': {'l': 15.54, 'q': 5, 'brand': 'Chex'},
    '568306986': {'l': 9.4, 'q': 4, 'brand': 'Chex'}
}

model_params = {
    'inc': 48,
    'nl': 5,
    'threshold': 4,
    'ur': 0.95,
    'time_limit': 1000*60*60 # 1 hour
}

inc = model_params['inc']
nl = model_params['nl']
threshold = model_params['threshold']
ratio = model_params['ur']
time_limit = model_params['time_limit']

sku_ids = list(skus_info.keys())
n = len(sku_ids)
id_idx_dict = dict(zip(sku_ids, range(n)))
idx_id_dict = dict(zip(range(n), sku_ids))

q = [skus_info[idx_id_dict[i]]['q'] for i in range(n)]
l = [skus_info[idx_id_dict[i]]['l'] for i in range(n)]

L = np.ceil(sum([q[i]*l[i] for i in range(n)])/ratio/inc/nl)*inc

# brand
brand_list = list(set([skus_info[key]['brand'] for key in skus_info.keys()]))
brand_sku_id_list = [[key for key in skus_info.keys() if skus_info[key]['brand'] == brand] for brand in brand_list]
brand_sku_id_list = [g for g in brand_sku_id_list if len(g)>=2]
brand_sku_idx_list = [[id_idx_dict[id] for id in brand_sku_id_list[i]] for i in range(len(brand_sku_id_list))]

u = [int(np.floor(inc/l[i])) for i in range(n)] # upper bound

O = [i for i in range(n) if q[i]*l[i] <= inc and q[i] < threshold] # put in single line


# initialize solver
try:
    solver = new_solver('sdo', integer=True)
except:
    warnings.warn("CPLEX is not found and switch to CBC!")
    solver = new_solver('sdo', integer=True, cplex=False)

solver.set_time_limit(time_limit)

# declarations
B2d = [[solver.IntVar(0, 1, f'B_{i}_{j}') for j in range(nl)] for i in range(n)]
Left = [[[solver.IntVar(0, 1, f'L_{i}_{ip}_{k}') for k in range(nl)] for ip in range(n)] for i in range(n)]
n2d = [[solver.IntVar(0, u[i], f'n_{i}_{k}') for k in range(nl)] for i in range(n)]
G2d = [[solver.IntVar(0, 1, f'G_{b}_{k}') for k in range(nl)] for b in range(len(brand_sku_idx_list))]

x = [[solver.NumVar(0.0, L, f'x_{i}_{k}') for k in range(nl)] for i in range(n)]
y = [[solver.NumVar(0.0, L, f'y_{i}_{k}') for k in range(nl)] for i in range(n)]
h = [[solver.NumVar(0.0, L, f'h_{i}_{k}') for k in range(nl)] for i in range(n)]
t = [[solver.NumVar(0.0, L, f't_{i}_{k}') for k in range(nl)] for i in range(n)]
o = [[solver.NumVar(0.0, L, f'o_{i}_{k}') for k in range(nl)] for i in range(n)]
d = [[solver.NumVar(0.0, L, f'd_{i}_{ip}') for ip in range(n)] for i in range(n)]
Y_max = solver.NumVar(0.0, L,f'Y_max')
Y = [solver.NumVar(0.0, L,f'Y_{k}') for k in range(nl)]
# constraints

# quantity
for i in range(n):
    solver.Add(sum([n2d[i][k] for k in range(nl)]) == q[i])

# no collisions
for i in range(n):
    for ip in range(n):
        for k in range(nl):
            if i != ip:
                solver.Add(Left[i][ip][k] + Left[ip][i][k] + (1 - B2d[i][k]) + (1 - B2d[ip][k]) >= 1)
                solver.Add(Left[i][ip][k] * 2 + Left[ip][i][k] * 2  <= B2d[i][k] + B2d[ip][k])
                solver.Add(y[i][k] + Left[i][ip][k] * L <= x[ip][k] + L)

# connected in one sku
for i in range(n):
    for k1 in range(nl):
        for k2 in range(nl):
            for k3 in range(nl):
                if k1 < k2 and k2 < k3:
                    solver.Add(B2d[i][k1] - B2d[i][k2] + B2d[i][k3] <= 1)

# connected in brand
for b in range(len(brand_sku_idx_list)):
    for k in range(nl):
        solver.Add(G2d[b][k] <= sum(B2d[i][k] for i in brand_sku_idx_list[b]))
        for i in brand_sku_idx_list[b]:
            solver.Add(G2d[b][k] >= B2d[i][k])

for b in range(len(brand_sku_idx_list)):
    for k1 in range(nl):
        for k2 in range(nl):
            for k3 in range(nl):
                if k1 < k2 and k2 < k3:
                    solver.Add(G2d[b][k1] - G2d[b][k2] + G2d[b][k3] <= 1)


# # connected in one brand
# for j in range(len(brand_sku_idx_list)):
#     bl = brand_sku_idx_list[j]
#     for i in bl:
#         for ip in bl:
#             for ipp in bl:
#                 if i < ip and ip < ipp:
#                     for k1 in range(nl):
#                         for k2 in range(nl):
#                             for k3 in range(nl):
#                                     solver.Add(B2d[i][k1] - B2d[ip][k2] + B2d[ipp][k3] <= 1)


for k in range(nl):
    solver.Add(Y_max >= Y[k])
    for i in range(n):
        solver.Add(Y[k] >= y[i][k])

# not split into two layers
for i in O:
    solver.Add(sum(B2d[i][k] for k in range(nl)) == 1)

# overlapping length (o)
for i in range(n):
    for k in range(nl):
        for kp in range(nl):
            solver.Add(t[i][k] - (1 - B2d[i][k]) * L <= y[i][kp] + (1 - B2d[i][kp]) * L)

for i in range(n):
    for k in range(nl):
        for kp in range(nl):
            solver.Add(h[i][k] + (1 - B2d[i][k]) * L >= x[i][kp] - (1 - B2d[i][kp]) * L)

# distance
for i in range(n):
    for ip in range(n):

        if i != ip:
            solver.Add(d[i][ip] == d[ip][i])
        else:
            solver.Add(d[i][ip]==0)

        for k in range(nl):
            if i != ip:
                solver.Add(x[ip][k] - y[i][k] <= d[i][ip] + (1 - Left[i][ip][k]) * L)

for i in range(n):
    for ip in range(n):
        for k in range(nl):
            for kp in range(nl):
                if i != ip:
                    solver.Add(x[i][k] - x[ip][kp] <= d[i][ip] + sum(Left[i][ip][kk] for kk in range(nl))*L + sum(Left[ip][i][kk] for kk in range(nl))*L)
                    solver.Add(-x[i][k] + x[ip][kp] >= -d[i][ip] - sum(Left[i][ip][kk] for kk in range(nl)) * L - sum(Left[ip][i][kk] for kk in range(nl)) * L)

for i in range(n):
    for k in range(nl):
        solver.Add(y[i][k] - x[i][k] == l[i] * n2d[i][k])
        solver.Add(o[i][k] == t[i][k] - h[i][k])
        # solver.Add(o[i][k] >= l_s[i] * B2d[i][k]) # must be connected
        solver.Add(y[i][k] <= L * B2d[i][k])
        solver.Add(y[i][k] >= t[i][k])
        solver.Add(t[i][k] >= h[i][k])
        solver.Add(h[i][k] >= x[i][k])
        solver.Add(B2d[i][k] <= n2d[i][k]) # enforce consistent

# objectives
solver.Maximize(sum([o[i][k]/L for i in range(n) for k in range(nl)])
                - sum([B2d[i][k] for i in range(n) for k in range(nl)])
                - sum(sum([d[i][ip]/L for i in g for ip in g]) for g in brand_sku_idx_list) \
                        /sum([sum([1 for i in g for ip in g]) for g in brand_sku_idx_list])
                - Y_max/L
                - .1*sum(Y[k]/L for k in range(nl))/nl
)

result_status = solver.Solve()

x = sol_val(x)
y = sol_val(y)
shelf = 0
n2d = sol_val(n2d)
B2d = sol_val(B2d)
d = sol_val(d)

result = copy.deepcopy(skus_info)
s = list(range(n))
x_layer = [[] for i in range(nl)]
y_layer = [[] for i in range(nl)]

for i in range(len(s)):  # i is local
    shelf_list = []
    layer_list = []
    x_list = []
    y_list = []
    n_list = []
    for k in range(nl):
        if B2d[i][k] > 0:
            if n2d[i][k] > 0:
                shelf_list.append(shelf)
                layer_list.append(k)
                x_list.append(x[i][k])
                y_list.append(y[i][k])
                n_list.append(n2d[i][k])
                x_layer[k].append(x[i][k])
                y_layer[k].append(y[i][k])
    result[idx_id_dict[s[i]]]['shelf'] = shelf_list
    result[idx_id_dict[s[i]]]['layer'] = layer_list
    result[idx_id_dict[s[i]]]['x'] = x_list
    result[idx_id_dict[s[i]]]['y'] = y_list
    result[idx_id_dict[s[i]]]['n'] = n_list

layout = create_layout_from_result(result, nl)
im = plot_layout(layout[0], shelf_length=int(L))
im.show()