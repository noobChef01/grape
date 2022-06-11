from ply import yacc
from lexer_files_knap.lexer import tokens, Lexer

import json

def read_weights(filename):
    with open(filename) as file:
        return json.load(file)

class WeightParser(object):

    tokens = tokens
    weights = read_weights("/mnt/d/college_notes/internship/grape/knapsack_ag/data/knapsack_weights.json")
    
    def __new__(cls, **kwargs):
        ## Does magic to allow PLY to do its thing.
        self = super(WeightParser, cls).__new__(cls, **kwargs)
        self.yacc = yacc.yacc(module=self,  tabmodule="knapsack_weight_parser_tab", **kwargs)
        return self.yacc

    def p_weight_char(self, p):
        '''exp : char'''
        p[0] = p[1]

    def p_weight_char_exp(self, p):
        '''exp : char exp''' 
        p[0] = p[1] + p[2]

    
    def p_weight_exp_char(self, p):
        '''exp : exp char'''
        p[0] = p[1] + p[2]

    def p_weight_symbol(self, p):
        '''char : SYMBOL'''
        p[0] = p[1][1] * self.weights[str(p[1][0])][1]

    
    # Error rule for syntax errors
    def p_error(self, p):
        print("Syntax error in input!")

def weight_parser(string):
    return WeightParser().parse(string, lexer=Lexer())

class ValueParser(object):

    tokens = tokens
    values = read_weights("/mnt/d/college_notes/internship/grape/knapsack_ag/data/knapsack_weights.json")
    
    def __new__(cls, **kwargs):
        ## Does magic to allow PLY to do its thing.
        self = super(ValueParser, cls).__new__(cls, **kwargs)
        self.yacc = yacc.yacc(module=self,  tabmodule="knapsack_value_parser_tab", **kwargs)
        return self.yacc

    def p_value_char(self, p):
        '''exp : char'''
        p[0] = p[1]

    def p_value_char_exp(self, p):
        '''exp : char exp''' 
        p[0] = p[1] + p[2]

    
    def p_value_exp_char(self, p):
        '''exp : exp char'''
        p[0] = p[1] + p[2]

    def p_value_symbol(self, p):
        '''char : SYMBOL'''
        p[0] = p[1][1] * self.values[str(p[1][0])][0]
    
    # Error rule for syntax errors
    def p_error(self, p):
        print("Syntax error in input!")

def value_parser(string):
    return ValueParser().parse(string, lexer=Lexer())

if __name__ == "__main__":
    weight = weight_parser("S10")
    value = value_parser("S10")
    print("Weight:", weight, "\tValue:", value)