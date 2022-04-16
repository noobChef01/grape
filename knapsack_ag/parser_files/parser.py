from ply import yacc
from lexer_files.lexer import tokens, Lexer
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

    def p_weight_bitString10(self, p):
        '''bitString10 : ZERO bitString9
            | ONE bitString9'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]
        
    def p_weight_bitString9(self, p):
        '''bitString9 : ZERO bitString8
            | ONE bitString8'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]

    def p_weight_bitString8(self, p):
        '''bitString8 : ZERO bitString7
            | ONE bitString7'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]

    def p_weight_bitString7(self, p):
        '''bitString7 : ZERO bitString6
            | ONE bitString6'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]

    def p_weight_bitString6(self, p):
        '''bitString6 : ZERO bitString5
            | ONE bitString5'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]

    def p_weight_bitString5(self, p):
        '''bitString5 : ZERO bitString4
            | ONE bitString4'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]

    def p_weight_bitString4(self, p):
        '''bitString4 : ZERO bitString3
            | ONE bitString3'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]


    def p_weight_bitString3(self, p):
        '''bitString3 : ZERO bitString2
                  | ONE bitString2'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2]

    def p_weight_bitString2(self, p):
        '''bitString2 : ZERO bit
                  | ONE bit'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][1]) + p[2] 

    def p_weight_bit(self, p):
        '''bit : ZERO
                  | ONE'''
        p[0] = p[1][0] * self.weights[str(p[1][1])][1]

    
    # Error rule for syntax errors
    def p_error(self, p):
        print("Syntax error in input!")

def weight_parser(string):
    return WeightParser().parse(string, lexer=Lexer())

class ValueParser(object):

    tokens = tokens
    weights = read_weights("/mnt/d/college_notes/internship/grape/knapsack_ag/data/knapsack_weights.json")
    
    def __new__(cls, **kwargs):
        ## Does magic to allow PLY to do its thing.
        self = super(ValueParser, cls).__new__(cls, **kwargs)
        self.yacc = yacc.yacc(module=self,  tabmodule="knapsack_value_parser_tab", **kwargs)
        return self.yacc

    def p_value_bitString10(self, p):
        '''bitString10 : ZERO bitString9
            | ONE bitString9'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]
        
    def p_value_bitString9(self, p):
        '''bitString9 : ZERO bitString8
            | ONE bitString8'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]

    def p_value_bitString8(self, p):
        '''bitString8 : ZERO bitString7
            | ONE bitString7'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]

    def p_value_bitString7(self, p):
        '''bitString7 : ZERO bitString6
            | ONE bitString6'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]

    def p_value_bitString6(self, p):
        '''bitString6 : ZERO bitString5
            | ONE bitString5'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]

    def p_value_bitString5(self, p):
        '''bitString5 : ZERO bitString4
            | ONE bitString4'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]

    def p_value_bitString4(self, p):
        '''bitString4 : ZERO bitString3
            | ONE bitString3'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]


    def p_value_bitString3(self, p):
        '''bitString3 : ZERO bitString2
                  | ONE bitString2'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2]

    def p_value_bitString2(self, p):
        '''bitString2 : ZERO bit
                  | ONE bit'''
        p[0] = (p[1][0] * self.weights[str(p[1][1])][0]) + p[2] 

    def p_value_bit(self, p):
        '''bit : ZERO
                  | ONE'''
        p[0] = p[1][0] * self.weights[str(p[1][1])][0]
    
    # Error rule for syntax errors
    def p_error(self, p):
        print("Syntax error in input!")

def value_parser(string):
    return ValueParser().parse(string, lexer=Lexer())

if __name__ == "__main__":
    weight = weight_parser("1011011010")
    value = value_parser("1011011010")
    print("Weight:", weight, "\tValue:", value)