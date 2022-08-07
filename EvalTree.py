import re
import json


class EvalTree(object):
    '''
    Base class for Tree Evaluation, which reads in the terminal node 
    definitions specified in a json file.
    '''

    def __init__(self, meta_file_path):
        self.terminal_node_meta = self.load_terminal_meta(meta_file_path)

    def load_terminal_meta(self, meta_file_path):
        with open(meta_file_path) as file:
            return json.load(file)


class EvalKnapSackTree(EvalTree):
    '''
    Evaluate a knapsack Parse Tree consisting of 0 and 1 bits at the 
    terminals.'''

    def __init__(self, w_threshold, meta_file_path):
        super().__init__(meta_file_path)
        self.w_threshold = w_threshold
        self.curr_weight = 0
        self.sorted_value_items = self.most_valuable()

    def most_valuable(self):
        value_weight_ratio = dict()
        for key in self.terminal_node_meta:
            value_weight_ratio[key] = self.terminal_node_meta[key]['value'] \
                / self.terminal_node_meta[key]['weight']
        return sorted(value_weight_ratio, key=lambda k: value_weight_ratio[k], reverse=True)

    def visit(self, tree, node_id):
        node = tree.get_node(node_id)
        if node.data.meta:
            bit = int(re.search(r'\d+', node.tag).group(0))
            weight = bit * node.data.meta['weight']
            self.curr_weight = self.curr_weight + weight
        for child in tree.children(node_id):
            self.visit(tree, child.identifier)

    def evaluate(self, tree):
        root_id = tree.root
        return self.visit(tree, root_id)

    def reset_params(self):
        self.curr_weight = 0

    def tree_meta_data(self, tree):
        self.reset_params()
        self.evaluate(tree)
        return self.curr_weight


class EvalSymRegTree(EvalTree):

    def __init__(self, meta_file_path):
        super().__init__(meta_file_path)
        # self.tree = tree

    def visit(self, tree, node_id):
        node = tree.get_node(node_id)
        if node.tag == 'DIVIDE':
            parent = self.tree.parent(node.identifier)
            children = tree.children(parent.identifier)
            for i, child in enumerate(children):
                if child.identifier == node.identifier:
                    tree.update_node(children[i+1].identifier, tag=children[i+1].tag + '+0.000001')
        for child in tree.children(node_id):
            self.visit(tree, child.identifier)

    def evaluate(self, tree):
        root_id = tree.root
        return self.visit(tree, root_id)

    # def tree_meta_data(self, tree):
    #     self.evaluate(tree)
    #     return self.curr_weight
