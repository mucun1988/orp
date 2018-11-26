#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:45:49 2018
@author: matthew.mu
"""

import sys, os

sys.path.extend([os.path.join('.', 'mbp')])

from mbp import MinBagPacker

weights = [1]*12 + [2]*5 + [3] + [11] + [100]
capacity = 12

packer = MinBagPacker(weights, capacity, dg=100)
packer.greedy_pack()
packer.bag_weights
packer.packing_result
packer.item_bag_result


weights = [8] + [6]*2
capacity = 10

packer = MinBagPacker(weights, capacity, dg=1)
packer.greedy_pack()
packer.bag_weights
packer.packing_result
packer.item_bag_result