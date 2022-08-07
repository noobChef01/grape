import re
import json
from treelib import Tree, Node


T_NT = 'NT'
T_T = 'T'


class Token(object):

    def __init__(self, type, name, meta=None) -> None:
        self.type = type
        self.name = name
        self.meta = meta

    def __str__(self) -> str:
        '''
        String representation of class instance of the format
        'ClassName(token-type, token-name, meta-data)'.

        Examples: 
            Token(NT, '<expr>', None)
            Token(T, 'X1', {"value": 10, "weight": 50})
        '''
        return f"Token({self.type}, {self.name}, {self.meta})"

    def __repr__(self) -> str:
        return self.__str__()


class ParseTree(object):

    def __init__(self, grammar, node_meta, type='knapsack'):
        self.type = type
        self.branch_idx = 0
        self.branch_depth = [0]
        self.start_rule = grammar.start_rule
        self.terminals = set(grammar.terminals)
        self.non_terminals = set(grammar.non_terminals)
        self.node_meta = node_meta
        self.reset_tree()

    def reset_tree(self):
        self.tree = Tree()
        self.make_root()

    def node_name(self, token_name):
        return re.sub(r'<|>', '', token_name)

    def make_token(self, token_name, token_idx=None):
        if token_name in self.terminals:
            if self.type == 'knapsack':
                return Token(T_T, token_name, meta=self.node_meta.get(f'{token_idx}'))
            else:
                return Token(T_T, token_name, meta=self.node_meta.get(token_name))
        else:
            return Token(T_NT, token_name, meta=self.node_meta.get(token_name))

    def make_root(self):
        root_name = self.node_name(self.start_rule)
        root_id = f"{root_name}_0_0"
        root_data = self.make_token(self.start_rule)
        self.tree.create_node(root_name.upper(),
                              root_id, data=root_data)
        self.next_expansion = root_id

    def set_expansion_node(self, idx):
        skip = 0
        leaves_sorted = sorted(
            self.tree.leaves(), key=lambda x: x.identifier.split("_")[-1])
        for i, node in enumerate(leaves_sorted):
            if node.data.type == T_T:
                skip += 1
                continue
            if int(node.identifier.split("_")[-1]) == idx:
                self.next_expansion = node.identifier
                self.branch_idx = i-skip
                break

    def update_tree(self, phenotype):
        # TODO: added for regression parse-tree
        # formatted_t = []
        # for t in self.terminals:
        #     match = re.match(r'x\[(\d+)\]', t)
        #     if match:
        #         formatted_t.append(r'x\[{}\]'.format(match.group(1)))
        #     else:
        #         formatted_t.append(t)
        # tokens = re.findall(
        #     r"<\w+>|\d+\.\d+|{}".format('|'.join(formatted_t)), phenotype)
        tokens = re.findall(
            r"<\w+>|\d+\.\d+|{}".format('|'.join(self.terminals)), phenotype)
        tokens = [self.make_token(tok, i+1) for i, tok in enumerate(tokens)]
        # TODO: branch depth could be a problem fix this
        self.branch_depth = self.branch_depth[:self.branch_idx] + [self.branch_depth[self.branch_idx]+1] * len(
            tokens) + self.branch_depth[self.branch_idx+1:]
        for i, token in enumerate(tokens):
            token_name = self.node_name(token.name)
            col = self.branch_idx + i
            depth = self.branch_depth[col]
            self.tree.create_node(token_name.upper(
            ), f'{token_name}_{depth}_{col}', parent=self.next_expansion, data=token)
        if len(tokens) > 1 and not self.tree.get_node(self.next_expansion).is_root():
            pid_exp_node = self.tree.parent(self.next_expansion).identifier
            for child in self.tree.children(pid_exp_node)[self.branch_idx+1:]:
                if not child.is_leaf():
                    for sub_child in self.tree.children(child.identifier):
                        old_id = sub_child.identifier
                        split_id = old_id.split('_')
                        old_col = split_id[-1]
                        new_id = "_".join(split_id[:-1]) + "_" + \
                            f'{(len(tokens)-1 + int(old_col))}'
                        self.tree.update_node(old_id, identifier=new_id)
                else:
                    old_id = child.identifier
                    split_id = old_id.split('_')
                    old_col = split_id[-1]
                    new_id = "_".join(split_id[:-1]) + "_" + \
                        f'{(len(tokens)-1 + int(old_col))}'
                    self.tree.update_node(old_id, identifier=new_id)

    # def collapse_tree(self):
    # TODO: old collapse parse_tree method
    #     if not self.tree.get_node(self.previous_expansion).is_root():
    #         subtree = self.tree.subtree(self.previous_expansion)
    #         children = subtree.children(subtree.root)
    #         self.lvl = 0
    #         self.tree = Tree()
    #         self.make_root()
    #         for col, child in enumerate(children):
    #             new_child_id = child.identifier.split('_')[0].strip()
    #             self.tree.create_node(
    #                 child.tag, f'{new_child_id}_{self.lvl}_{col}', parent=self.next_expansion, data=child.data)
    #         self.lvl += 1
    #         self.previous_expansion = self.next_expansion
    #         self.next_expansion = self.tree.children(self.next_expansion)[
    #             self.branch_idx].identifier

    def display(self, property=None):
        if property:
            return self.tree.show(data_property=property)
        return self.tree.show()

class ParseTree():

    def __init__(self, grammar, node_meta):
        self.start_rule = grammar.start_rule
        self.terminals = set(grammar.terminals)
        self.non_terminals = set(grammar.non_terminals)
        self.node_meta = node_meta
        self.expandable_nodes = []
        self.set_root()

    def strip(self, non_terminal):
        return re.sub(r'<|>', '', non_terminal)

    def get_token(self, token_name):
        type = T_T if token_name in self.terminals else T_NT
        return Token(type, token_name.upper(), meta=self.node_meta.get(f'{token_name}'))

    def set_root(self):
        self.tree = Tree()
        root_name = self.strip(self.start_rule)
        rid = root_name + '_0_0'
        root_meta = self.get_token(root_name)
        self.tree.create_node(root_name.upper(),
                              rid, data=root_meta)
        self.expandable_nodes.append([rid, 0])
        self.set_expansion_node(0)

    def has_binary_op(self, tokens):
        # TODO: in-case of inline function: eg: add(x, y)
        # def has_binary_op(self, tokens, chosen_prod):
        # if chosen_prod and re.search(r'\(.+\)', chosen_prod):
        #     return True, None
        for i, t in enumerate(tokens):
            meta = self.node_meta.get(t)
            if meta and t not in self.terminals \
                    and meta.get('type') == 'operator' and int(meta.get('arity')) == 2:
                return True, (i-1, i, i+1)
        return False, None

    def increment_node_ids(self, expandable_nodes):
        result = []
        for node in expandable_nodes:
            old_id_split = node[0].split('_')
            new_idx = int(old_id_split[-1])+2
            new_id = '_'.join(old_id_split[:-1]) + "_" + str(new_idx)
            self.tree.update_node(node[0], identifier=new_id)
            result.append([new_id, new_idx])
        return result

    def add_binary_operation(self, tokens, indices):
        lvl = self.tree.level(self.to_expand)
        l_idx, op_idx, r_idx = indices
        left_node = Node(
            tag=tokens[l_idx].upper(),
            identifier=f'{tokens[l_idx]}_left_{lvl+2}_{self.branch_idx}',
            data=self.get_token(tokens[l_idx])
        )
        op_tag = tokens[op_idx].upper()
        op_identifier = f'{tokens[op_idx]}_{lvl+1}_{self.branch_idx+1}'
        op_data = self.get_token(tokens[op_idx])
        right_node = Node(
            tag=tokens[r_idx].upper(),
            identifier=f'{tokens[r_idx]}_right_{lvl+2}_{self.branch_idx+2}',
            data=self.get_token(tokens[r_idx])
        )
        self.tree.update_node(self.to_expand, tag=op_tag,
                              identifier=op_identifier,
                              data=op_data)
        self.tree.add_node(left_node, parent=op_identifier)
        self.tree.add_node(right_node, parent=op_identifier)

        self.expandable_nodes = self.expandable_nodes[:self.branch_idx] \
            + [[left_node.identifier, self.branch_idx],
                [op_identifier, self.branch_idx+1],
                [right_node.identifier, self.branch_idx+2]] \
            + self.increment_node_ids(self.expandable_nodes[self.branch_idx+1:])

    def grow(self, chosen_prod):
        # TODO: fix to single constant node
        tokens = re.findall(
            r"<\w+>|x\[\d+\]|{}".format('|'.join(self.terminals)), chosen_prod)
        tokens = [self.strip(t) for t in tokens]
        flag, indices = self.has_binary_op(tokens)
        if flag:
            self.add_binary_operation(tokens, indices)
        else:
            node_name = self.strip(tokens[0])
            node_data = self.get_token(node_name)
            lvl = self.tree.level(self.to_expand)
            new_id = f'{node_name}_{lvl}_{self.branch_idx}'
            self.tree.update_node(self.to_expand, tag=node_name.upper(
            ), identifier=new_id, data=node_data)
            if node_name in self.terminals:
                _ = self.expandable_nodes.pop(self.branch_idx)
            else:
                self.expandable_nodes[self.branch_idx] = [
                    new_id, self.branch_idx]

    def set_expansion_node(self, idx):
        self.to_expand, self.branch_idx = self.expandable_nodes[idx]

    def display(self, property=None):
        if property:
            return self.tree.show(data_property=property)
        return self.tree.show()
