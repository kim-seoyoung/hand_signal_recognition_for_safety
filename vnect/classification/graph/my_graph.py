import sys

sys.path.extend(['../'])
from graph import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0,16),(16,1),(1,2),(2,3),
                    (3,4),(4,17),(1,5),(5,6),
                    (6,7),(7,18),(1,15),(15,14),
                    (14,8),(8,9),(9,10),(10,19),
                    (14,11),(11,12),(12,13),(13,20)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
