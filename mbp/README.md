# Minimum Bag Problem (mbp)

## Problem Statement
Given n items with weights {w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>} and bags with capacity W, the minimum bag problem aims to put all n items into bags in a way such that the number of bags to be used is as small as possible.

## Methodology 
The package is built upon `Google Optimization Tools` (`OR-Tools`). So the [OR-Tools](https://developers.google.com/optimization/) from google needs to be installed first. The instructions to install `OR-Tools` on GNU-Linux, Macos and Windows are well-documented [here](https://developers.google.com/optimization/install/python/).




## How to use `mbp`?
Let us go through an example to illustrate how to use the package.

Suppose we have 20 items with weights specified as 
```
weights = [1]*12 + [2]*5 + [3] + [11] + [100]
```
and the capacity of each bag is
```
capacity = 12
```

We can solve the minimum bag problem by simply leveraging the `MinBagPacker` class:
```
from mbp import MinBagPacker
packer = MinBagPacker(weights, capacity)
packer.greedy_pack()
```

The solution can be by fetching ```packer.packing_result```, which outputs
```
[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 0, -1]
```
indicating the items are packed into which bag. As reflected from the output, 3 boxes are used to pack the 20 items. The last item has `-1` meaning that we cannot pack this item into any bag, since its weight alone exceeds the bag capacity. 
