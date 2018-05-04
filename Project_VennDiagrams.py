# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:17:56 2018

@author: lbnal
"""
from matplotlib_venn import venn3, venn2
from matplotlib import pyplot

all_kmeans_feats = set([7, 10, 12, 13, 15, 20, 32, 39, 46, 54, 56, 57, 65, 69, 76])

avg_kmeans_feats = set([0, 6, 7, 12, 15, 17, 18, 20, 21, 23, 27, 30, 31, 33, 36,
                        37, 41, 42, 44, 46, 47, 59, 70, 71, 75, 76])

tree_feats = set([0, 3, 5, 7, 10, 13, 17, 20, 30, 32, 33, 37, 39, 41, 42, 46, 48,
              52, 53, 54, 55, 56, 60, 62, 63, 65, 66])

ann_feats = set([0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22,
                 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41,
                 44, 47, 48, 49, 52, 55, 56, 58, 62, 63, 64, 66, 67, 69, 76])

forest_feats = set([0, 1, 2, 7, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 30, 32,
                    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48, 49,
                    50, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    68, 69, 70, 72, 73, 75, 76])

fig = pyplot.figure(figsize=(8,6))
venn3([tree_feats, all_kmeans_feats, ann_feats], set_labels = ('Tree [27]', 'K Means [15]', 'ANN [49]'))
pyplot.show()

fig = pyplot.figure(figsize=(8,6))
venn3([tree_feats, avg_kmeans_feats, ann_feats], set_labels = ('Tree [27]', 'Avg K Means [26]', 'ANN [49]'))
pyplot.show()

fig = pyplot.figure(figsize=(8,6))
venn3([tree_feats, forest_feats, ann_feats], set_labels = ('Tree [27]', 'Forest[53]', 'ANN [49]'))
pyplot.show()

fig = pyplot.figure(figsize=(8,6))
venn2([tree_feats, forest_feats], set_labels = ('Tree [27]', 'Forest[53]'))
pyplot.show()