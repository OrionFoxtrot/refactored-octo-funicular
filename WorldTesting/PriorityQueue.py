#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Lab 2
# Spring 2024, Swarthmore College
########################################

class PriorityQueue(object):
    def __init__(self):
        """
        Represent the min priority queue as a heap, which is a complete
        binary tree represented as a list.

        A user of the PQ should only call the methods: isEmpty,
        getSize, insert, and remove. All other methods are for internal
        use only.
        """
        self.heap = []
    def isEmpty(self):
        """Returns True of the PQ is empty, otherwise False"""
        return len(self.heap) == 0
    def getSize(self):
        """Returns the integer size of the PQ"""
        return len(self.heap)
    def insert(self, priority, item):
        """Inserts the item with given priority"""
        self.heap.append((priority, item))
        self._bubbleUp(self.getSize()-1)
    def remove(self):
        """Removes the item in the PQ with the minimum priority"""
        if self.getSize() == 0:
            raise RuntimeError("can't remove from an empty PQ")
        removed_item = self.heap[0][1]
        last_index = self.getSize()-1
        self.heap[0] = self.heap[last_index]
        self.heap.pop(last_index)
        self._bubbleDown(0)
        return removed_item
    def _left(self, index):
        return 2*index + 1
    def _right(self, index):
        return 2*index + 2
    def _parent(self, index):
        return (index-1)//2
    def _bubbleUp(self, index):
        if index != 0:
            p_index = self._parent(index)
            if self.heap[index][0] < self.heap[p_index][0]:
                self._swap(index, p_index)
                self._bubbleUp(p_index)
    def _bubbleDown(self, index):
        l = self._left(index)
        if l < self.getSize():
            # has a left child
            minIndex = l
            r = self._right(index)
            if r < self.getSize():
                # also has a right child, see if its priority is lower
                if self.heap[r][0] < self.heap[l][0]:
                    minIndex = r
            if self.heap[minIndex][0] < self.heap[index][0]:
                self._swap(index, minIndex)
                self._bubbleDown(minIndex)
    def _swap(self, i1, i2):
        self.heap[i1], self.heap[i2] = self.heap[i2], self.heap[i1]