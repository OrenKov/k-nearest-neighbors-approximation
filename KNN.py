import numpy as np
# **** Numpy Usage *** #
# (*) np.array - being used for 'smart' indexing and for transposing an array efficiently.
# (*) np.average - being used to find the average of a list efficiently

from scipy.spatial import distance
# **** Scipy Usage *** #
# (*) cdist - a built-in function to measure euclidean distance in the space.

import heapq
# Being used to maintain the best KNN while traversing the tree, efficiently.


class Node:
    """
    A node of a KDTree.
    contains relevant data such as split_axis and split_value of a node.
    """
    def __init__(self, left=None, right=None, split_axis=None, split_val=None, values=None):
        self.left = left
        self.right = right
        self.split_axis = split_axis  # in our case: [0,1,2] for [x,y,z]
        self.split_val = split_val
        self.values = values          # Only leaves should have values different than None.


class KDTree:
    """
    A k-d-tree, created by a given set of points of any dimension D, and can be used to find the KNN of a point
    in of dimension D.
    """
    def __init__(self, points, k):
        """
        :param points: a list of points of the same dimension (dim).
        :param k: number of approximated nearest neighbours of a point the tree can find efficiently.
        :param dim: the dimension of the points the tree holds. 3 by default.
        """
        if not self._check_points():
            return
        self.points = list(set(points))
        self.dim = len(self.points[0])
        self.len = len(self.points)
        self.k = k
        if not self._check_k():
            return
        self.tree = self._build_tree(points=self.points, dim=self.dim, i=0)

    # ************ #
    #      API     #
    # ************ #
    def knn(self, p):
        """
        Gets the KNN of the given point p, using the KDTree that was built.
        Cannot be used if the function "build_tree" wasn't called.
        :param p: a given point of dimension self.dim.
        :return:
        """
        return np.array(self._knn(p=p, k=self.k, current_node=self.tree), dtype=tuple)[:, 1].tolist()

    def get_z_avg(self, p):
        """
        Return the average value of 'z' axis in a given point containing segment.
        :param p: The given point.
        :return:  The average value of 'z' axis in the point's segment.
        """
        return self._get_axis_avg(p, 2)

    # ************ #
    #   HELPERS    #
    # ************ #

    def _build_tree(self, points, dim, i=0):
        """
        Builds a dim-d-tree out of the given list of points.
        :param points: A list of points
        :param dim: The dimension
        :param i:
        :return:
        """
        self.tree = Node()

        # Grow the trunk of the tree:
        if len(points) > self.k:
            try:
                sorted_points = sorted(points, key=lambda x: x[i])
            except IndexError as e:
                print(f"Please make sure all the points in the list are from dimension {dim} ", "\nError: ", e)
                return
            i = (i + 1) % dim
            median_index = len(sorted_points) // 2
            node = Node(
                left=self._build_tree(sorted_points[: median_index], dim, i),  # < of the median
                right=self._build_tree(sorted_points[median_index:], dim, i),  # >= of the median
                split_val=sorted_points[median_index],
                split_axis=i
            )
            return node

        # Grow a Leaf: 0 <= [Amount of points in a segment] <= self.k
        else:
            return Node(values=points)

    def _knn(self, p, k, current_node=None, heap=None):
        """
        given a point (x,y,z) in the data, returns its K ** approximated ** nearest neighbors.
        :param p: An input point.
        :param current_node: The node being checked.
        :param k: The number of NN. as for now, being used only with the K of the tree, but can be used with other K as
                    well.
        :return: a list of the k-nn of p.
        """
        if heap is None:
            heap = []

        # If we made it to a leaf, add it's relevant points to the heap:
        if current_node.values:
            self._add_leaf_points_to_heap(p, current_node, heap, k)
            return

        # If it is not a leaf, traverse to the best-approximated leaves:
        self._traverse_to_best_leaves(p, current_node, k, heap)
        return heap

    def _add_leaf_points_to_heap(self, p, current_node, heap, k):
        """
        Adding the node.values of a node to the given heap. the number of items in the heap is limited to k.
        NOTE: Using Euclidean distance as measure of distance.
        :param current_node: A leaf in the KDTree
        :param heap: The heap of the values.
        :return:
        """
        distances = distance.cdist(current_node.values, [p]).flatten()
        neighbours = np.array((distances, current_node.values), dtype=list).T  # array of [[point1, distance1], ..]
        for point in neighbours:
            if len(heap) < k:
                heapq.heappush(heap, tuple(point))  # (dist, [x,y,z] of current_node) is pushed
            elif point[0] < -heap[0][0]:
                heapq.heappushpop(heap, point)

    def _traverse_to_best_leaves(self, p, current_node, k, heap):
        """
        Traversing the KDTree for finding the best leaf for the point p.
        :param p: A given point.
        :param current_node: The current node in the tree that is traversed.
        :param k: Number of approximated NN needed to be found in the tree.
        :param heap: The heap that keeps the KNN.
        :return:
        """
        axis = current_node.split_axis
        if p[axis] < current_node.split_val[axis]:
            self._knn(p, k, current_node.left, heap)
            if len(heap) < k:
                self._knn(p, k, current_node.right, heap)
        else:
            self._knn(p, k, current_node.right, heap)
            if len(heap) < k:
                self._knn(p, k, current_node.left, heap)

    def _get_axis_avg(self, p, axis):
        """
        Generalized method to find the average of an axis value in a specific point segment.
        :param p: The point
        :param axis: The axis to preform average on.
        :return: The average
        """
        if axis > len(self.points[0]):
            return "Your dimension is high in the sky"
        return np.average(np.array(self._find_leaf(p, self.tree).values)[:, axis])

    def _find_leaf(self, p, current_node):
        """
        Finds the appropriate segment (leaf) of a point.
        :return: The leaf (Node type) fit to the point.
        """
        if current_node.values:
            return current_node

        axis = current_node.split_axis
        if p[axis] < current_node.split_val[axis]:
            return self._find_leaf(p, current_node.left)
        else:
            return self._find_leaf(p, current_node.right)

    # ************ #
    #   CHECKS     #
    # ************ #
    def _check_points(self):
        """
        Makes sure the points input is valid.
        :param points: suppose to be a list of points.
        :return:
        """
        if type(self.points) is not list or len(self.points) < 1:
            print("Please insert a list containing at least one point as an input.")
            return False
        return True

    def _check_k(self):
        """
        Makes sure K is large enough compared to the points input list length.
        :param len: The length of the
        :param k:
        :return:
        """
        if self.len < self.k:
            print("Please choose a K that is at lease the size of the points list.")
            return False
        elif type(self.k) is not int:
            print("Please insert K as an integer.")
            return False
        return True
