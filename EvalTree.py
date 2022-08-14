import re
import json


class EvalTree(object):
    '''
    Base class for Tree Evaluation, which reads in the terminal node
    definitions specified in a json file.
    '''

    def __init__(self, meta_file_path):
        self.node_meta = self.load_terminal_meta(meta_file_path)

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

    def __init__(self, meta_file_path, bnf_grammar):
        super().__init__(meta_file_path)
        self.grammar = bnf_grammar

    def reset(self, tree):
        self.tree = tree
        self.is_valid = True

    def get_visitor_method(self, node_type):
        return {
            'operator': 'visit_op_node_T',
            'decimal': 'visit_decimal_node_NT',
            'constant': 'visit_constant_node_T',
            'non_terminal': 'visit_NT',
            'input': 'visit_input_T',
            # 'epsilon': 'visit_epsilon_T'
        }[node_type]

    def set_method(self, node):
        method_name = ''
        if node.tag == '.':
            method_name = self.get_visitor_method('decimal')
        elif node.data.type == 'NT' or node.tag == 'op':
            method_name = self.get_visitor_method('non_terminal')
        elif node.data.name in self.grammar.terminals:
            if node.data.meta['type'] == 'operator':
                method_name = self.get_visitor_method('operator')
            # elif node.data.meta['type'] == 'epsilon':
            #     method_name = self.get_visitor_method('epsilon')
            elif node.data.meta['type'] == 'constant':
                method_name = self.get_visitor_method('constant')
            else:
                method_name = self.get_visitor_method('input')
        return method_name

    def visit(self, node):
        method_name = self.set_method(node)
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(
            'No visit method defined for node: {}'.format(node.tag))

    # def visit_epsilon_T(self, node):
    #     return 1e-8

    def visit_op_node_T(self, node):
        if node.data.meta.get('op_value') == '-':
            left, right = self.tree.children(node.identifier)
            l_val = self.visit(left)
            r_val = self.visit(right)
            # if l_val == r_val:
            #     self.is_valid = False
            #     return None
            try:
                return l_val - r_val
            except TypeError:
                pass
                # print("Nodes not fully expanded")

        elif node.data.meta.get('op_value') == '+':
            left, right = self.tree.children(node.identifier)
            try:
                return self.visit(left) + self.visit(right)
            except TypeError:
                pass
                # print("Nodes not fully expanded")

        elif node.data.meta.get('op_value') == '*':
            left, right = self.tree.children(node.identifier)
            try:
                return self.visit(left) * self.visit(right)
            except TypeError:
                pass
                # print("Nodes not fully expanded")

        elif node.data.meta.get('op_value') == '/':
            left, right = self.tree.children(node.identifier)
            l_val = self.visit(left)
            r_val = self.visit(right)
            if r_val == 0:
                self.is_valid = False
                return None
            try:
                return l_val / r_val
            except TypeError:
                pass
                # print("Nodes not fully expanded")

    def visit_decimal_node_NT(self, node):
        val = ''
        for i, child in enumerate(self.tree.children(node.identifier)):
            if i == 2:
                val += '.'
            val += str(self.visit(child))
        try:
            return float(val)
        except ValueError:
            pass
            # print("Nodes not fully expanded")

    def visit_constant_node_T(self, node):
        return str(node.data.meta.get('value'))

    def visit_input_T(self, node):
        # TODO: read from X-train and use a random value
        return 1

    def visit_NT(self, node):
        for child in self.tree.children(node.identifier):
            self.visit(child)

    def evaluate(self, tree):
        self.reset(tree)
        tree_val = self.visit(self.tree.get_node(tree.root))
        return self.is_valid
