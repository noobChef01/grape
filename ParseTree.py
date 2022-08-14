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


class ParseTree():

    def __init__(self, grammar, node_meta):
        self.start_rule = grammar.start_rule
        self.terminals = set(grammar.terminals)
        self.non_terminals = set(grammar.non_terminals)
        self.type = node_meta["type"]
        self.node_meta = node_meta["meta"]
        self.expandable_nodes = []
        self.set_root()

    def strip(self, non_terminal):
        return re.sub(r'<|>', '', non_terminal)

    def get_token(self, token_name):
        token_type = T_T if token_name in self.terminals else T_NT
        if self.type == 'knapsack' and token_type == T_T:
            # knapsack item numbering starts from 1
            return Token(token_type, token_name, meta=self.node_meta.get(f'{self.branch_idx+1}'))
        return Token(token_type, token_name, meta=self.node_meta.get(f'{token_name}'))

    def set_root(self):
        self.tree = Tree()
        root_name = self.strip(self.start_rule)
        rid = root_name + '_0_0'
        root_meta = self.get_token(root_name)
        self.tree.create_node(root_name,
                              rid, data=root_meta)
        self.expandable_nodes.append([rid, 0])
        self.set_expansion_node(0)

    def has_binary_op(self, tokens):
        # match = re.match(r'x\[(\d+)\]', t)
        # formatted_t.append(r'x\[{}\]'.format(match.group(1)))
        # TODO: in-case of inline function: eg: add(x, y)
        # def has_binary_op(self, tokens, chosen_prod):
        # if chosen_prod and re.search(r'\(.+\)', chosen_prod):
        #     return True, None
        for i, t in enumerate(tokens):
            if i > 5:
                break
            meta = self.node_meta.get(t)
            if meta and t not in self.terminals \
                    and meta.get('type') == 'operator' and int(meta.get('arity')) == 2:
                return True, (i-1, i, i+1)
        return False, None

    def increment_node_ids(self, expandable_nodes, n_added):
        result = []
        for node in expandable_nodes:
            old_id_split = node[0].split('_')
            new_idx = int(old_id_split[-1])+(n_added-1)
            new_id = '_'.join(old_id_split[:-1]) + "_" + str(new_idx)
            self.tree.update_node(node[0], identifier=new_id)
            result.append([new_id, new_idx])
        return result

    def add_binary_operation(self, tokens, indices):
        lvl = self.tree.level(self.to_expand)
        l_idx, op_idx, r_idx = indices
        left_node = Node(
            tag=tokens[l_idx],
            identifier=f'{tokens[l_idx]}_left_{lvl+2}_{self.branch_idx}',
            data=self.get_token(tokens[l_idx])
        )
        op_tag = tokens[op_idx]
        op_identifier = f'{tokens[op_idx]}_{lvl+1}_{self.branch_idx+1}'
        op_data = self.get_token(tokens[op_idx])
        right_node = Node(
            tag=tokens[r_idx],
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
            + self.increment_node_ids(self.expandable_nodes[self.branch_idx+1:], 3)

    def contains_constant(self, tokens):
        for token in tokens:
            if token == '.':
                return True
        return False

    def grow(self, chosen_prod):
        tokens = re.findall(
            r"<\w+>|x\[\d+\]|\.|{}".format('|'.join(self.terminals)), chosen_prod)
        tokens = [self.strip(t) for t in tokens]
        flag, indices = self.has_binary_op(tokens)
        if flag:
            self.add_binary_operation(tokens, indices)
        elif len(tokens) > 1:
            if self.type != 'knapsack' and self.contains_constant(tokens):
                lvl = self.tree.level(self.to_expand)
                decimal_nid = f'._{lvl}_{self.branch_idx}'
                self.tree.update_node(self.to_expand,
                                      tag='.', identifier=decimal_nid)
                temp = []
                i = 0
                for token in tokens:
                    if token == '.':
                        continue
                    node_name = self.strip(token)
                    node_data = self.get_token(node_name)
                    lvl = self.tree.level(decimal_nid)
                    node_id = f'{node_name}_{lvl+1}_{i}'
                    self.tree.create_node(
                        tag=node_name, identifier=node_id, data=node_data, parent=decimal_nid)
                    temp.append([node_id, i])
                    i += 1
                self.expandable_nodes = self.expandable_nodes[:self.branch_idx] \
                    + [[e_node[0], self.branch_idx+e_node[1]] for e_node in temp] \
                    + self.increment_node_ids(self.expandable_nodes[self.branch_idx+1:], len(temp))
            else:
                temp = []
                for i, token in enumerate(tokens):
                    node_name = self.strip(token)
                    node_data = self.get_token(node_name)
                    lvl = self.tree.level(self.to_expand)
                    node_id = f'{node_name}_{lvl}_{i}'
                    self.tree.create_node(
                        tag=node_name, identifier=node_id, data=node_data, parent=self.to_expand)
                    temp.append([node_id, i])
                self.expandable_nodes = temp
                self.set_expansion_node(0)
        else:
            node_name = self.strip(tokens[0])
            node_data = self.get_token(node_name)
            lvl = self.tree.level(self.to_expand)
            new_id = f'{node_name}_{lvl}_{self.branch_idx}'
            self.tree.update_node(
                self.to_expand, tag=node_name, identifier=new_id, data=node_data)
            self.expandable_nodes[self.branch_idx] = [
                new_id, self.branch_idx]

    def set_expansion_node(self, idx):
        if idx <= len(self.expandable_nodes)-1:
            self.to_expand, self.branch_idx = self.expandable_nodes[idx]

    def display(self, property=None):
        if property:
            return self.tree.show(data_property=property)
        return self.tree.show()
