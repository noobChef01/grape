import re
import json


class EvalTree(object):

    def __init__(self, w_threshold, meta_file_path) -> None:
        super().__init__()
        self.w_threshold = w_threshold
        self.value = None
        self.weight = None
        self.meta_data = self.load_meta_data(meta_file_path)
        self.sorted_value_items = self.most_valuable()

    def load_meta_data(self, meta_file):
        with open(meta_file) as file:
            return json.load(file)

    def most_valuable(self):
        value_weight_ratio = dict()
        for key in self.meta_data:
            value_weight_ratio[key] = self.meta_data[key]['value'] / \
                self.meta_data[key]['weight']
        return sorted(value_weight_ratio, key=lambda k: value_weight_ratio[k], reverse=True)

    def visit(self, tree, node_id):
        node = tree.get_node(node_id)
        if node.data.meta:
            bit = int(re.search(r'BIT:(\d+)_', node.tag).group(1))
            value, weight = bit * \
                node.data.meta['value'], bit*node.data.meta['weight']
            self.value = (self.value if self.value else 0) + value
            self.weight = (self.weight if self.weight else 0) + weight
        for child in tree.children(node_id):
            self.visit(tree, child.identifier)

    def evaluate(self, tree):
        root_id = tree.root
        return self.visit(tree, root_id)

    def reset_params(self):
        self.value = None
        self.weight = None

    def tree_meta_data(self, tree):
        self.reset_params()
        self.evaluate(tree)
        return self.value, self.weight
