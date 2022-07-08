import re
import json
from Token import (T_NT, T_T, Token)
from treelib import Node, Tree


class ParseTree(object):

    def __init__(self, grammar, meta_file_path):
        self.lvl = 0
        self.expansion_idx = 0
        self.start_rule = grammar.start_rule
        self.terminals = set(grammar.terminals)
        self.non_terminals = set(grammar.non_terminals)
        self.meta_data = self.load_meta_data(meta_file_path)
        self.tree = Tree()
        self.make_root()

    def load_meta_data(self, meta_file):
        with open(meta_file) as file:
            return json.load(file)

    def node_name(self, token_name):
        return re.sub(r'<|>', '', token_name)

    def make_token(self, token_name, token_idx=None):
        if token_name in self.terminals:
            return Token(T_T, f'bit:{token_name}_pos:{token_idx}', meta=self.meta_data.get(f'{token_idx}'))
        else:
            return Token(T_NT, token_name, meta=self.meta_data.get(token_name))

    def make_root(self):
        root_name = self.node_name(self.start_rule)
        root_id = f"{root_name}_{self.lvl}"
        root_data = self.make_token(self.start_rule)
        self.tree.create_node(root_name.upper(),
                              root_id, data=root_data)
        self.next_expansion = root_id
        self.previous_expansion = root_id
        self.lvl += 1

    def expansions_available(self):
        return self.tree.children(self.previous_expansion)

    def set_expansion_node(self, idx):
        self.expansion_idx = idx
        if idx != 0:
            self.previous_expansion = self.tree.parent(
                self.next_expansion).identifier
            if idx <= len(self.expansions_available())-1:
                self.next_expansion = self.expansions_available()[
                    self.expansion_idx].identifier

    def update_tree(self, phenotype):
        tokens = re.findall(f"<\w+>|{'|'.join(self.terminals)}", phenotype)
        tokens = [self.make_token(tok, i+1) for i, tok in enumerate(tokens)]
        for col, token in enumerate(tokens):
            token_name = self.node_name(token.name)
            self.tree.create_node(token_name.upper(
            ), f'{token_name}_{self.lvl}_{col}', parent=self.next_expansion, data=token)
        self.lvl += 1
        self.previous_expansion = self.next_expansion
        self.next_expansion = self.tree.children(self.next_expansion)[
            self.expansion_idx].identifier

    def collapse_tree(self):
        if not self.tree.get_node(self.previous_expansion).is_root():
            subtree = self.tree.subtree(self.previous_expansion)
            children = subtree.children(subtree.root)
            self.lvl = 0
            self.tree = Tree()
            self.make_root()
            for col, child in enumerate(children):
                new_child_id = child.identifier.split('_')[0].strip()
                self.tree.create_node(
                    child.tag, f'{new_child_id}_{self.lvl}_{col}', parent=self.next_expansion, data=child.data)
            self.lvl += 1
            self.previous_expansion = self.next_expansion
            self.next_expansion = self.tree.children(self.next_expansion)[
                self.expansion_idx].identifier

    def display(self, property=None):
        if property:
            return self.tree.show(data_property=property)
        return self.tree.show()
