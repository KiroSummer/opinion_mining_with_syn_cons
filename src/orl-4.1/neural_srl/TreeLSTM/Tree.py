class Tree(object):
    def __init__(self, index):
        self.parent = None
        self.is_left = False
        self.index = index
        self.left_children = list()
        self.left_num = 0
        self.right_children = list()
        self.right_num = 0
        self._depth = -1
        self.order = []

    def add_left(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        child.is_left = True
        self.left_children.append(child)
        self.left_num += 1

    def add_right(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        child.is_left = False
        self.right_children.append(child)
        self.right_num += 1

    def size(self):  # compute the total size of the Tree
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.left_num):
            count += self.left_children[i].size()
        for i in range(self.right_num):
            count += self.right_children[i].size()
        self._size = count
        return self._size

    def depth(self):  # compute the depth of the Tree
        if self._depth > 0:
            return self._depth
        count = 0
        if self.left_num + self.right_num > 0:
            for i in range(self.left_num):
                child_depth = self.left_children[i].depth()
                if child_depth > count:
                    count = child_depth
            for i in range(self.right_num):
                child_depth = self.right_children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def traverse(self):  # traverse the Tree
        if len(self.order) > 0:
            return self.order

        for i in range(self.left_num):
            left_order = self.left_children[i].traverse()
            self.order.extend(left_order)
        for i in range(self.right_num):
            right_order = self.right_children[i].traverse()
            self.order.extend(right_order)
        self.order.append(self.index)  # append the root
        return self.order


def creatTree(heads):
    tree = []
    # current sentence has already been numberized [form, head, rel]
    root = None
    for idx, head in enumerate(heads):
        tree.append(Tree(idx))

    for idx, head in enumerate(heads):
        if head == -1:  # -1 mszhang, 0 kiro
            root = tree[idx]
            continue
        if head < 0:
            print('error: multi roots')
        if head > idx:
            tree[head].add_left(tree[idx])
        if head < idx:
            tree[head].add_right(tree[idx])
        if head == idx:
            print('error: head is it self.')

    return root, tree
