from ply import yacc

from lexer import tokens, Lexer

# class Node:
#     def __init__(self,type,children=None,leaf=None):
#         self.type = type
#         if children:
#              self.children = children
#         else:
#              self.children = [ ]
#         self.leaf = leaf

class Parser(object):

    tokens = tokens
    # Parsing rules
    precedence = (
        ('left','PLUS','MINUS'),
        ('left','TIMES','DIVIDE'),
        # ('right','UMINUS'),
        )
    # # dictionary of names
    # names = { }
    
    def __new__(cls, **kwargs):
        ## Does magic to allow PLY to do its thing.
        self = super(Parser, cls).__new__(cls, **kwargs)
        self.yacc = yacc.yacc(module=self,  tabmodule="attrgram_parser_tab", **kwargs)
        return self.yacc


    # def p_statement_assign(self, t):
    #     'statement : NAME EQUALS expression'
    #     self.names[t[1]] = t[3]

    # def p_statement_expr(self, t):
    #     'statement : expression'
    #     print(t[1])

    def p_expression_binop(self, t):
        '''expression : expression PLUS expression
                    | expression MINUS expression
                    | expression TIMES expression
                    | expression DIVIDE expression'''
        if t[2] == '+'  : t[0] = t[1] + t[3]
        elif t[2] == '-': t[0] = t[1] - t[3]
        elif t[2] == '*': t[0] = t[1] * t[3]
        elif t[2] == '/': t[0] = t[1] / t[3]
            # if t[3] == 0:
            #     print("parse tree not in grammar")
            # else: 
            #     t[0] = t[1] / t[3]
        # t[0] = Node("binop", [t[1],t[3]], t[2])

    # def p_expression_uminus(self, t):
    #     # 'expression : MINUS expression %prec UMINUS'
        
    #     t[0] = -t[2]
    #     # t[0] = Node("uminus", [t[2]], "-")

    def p_expression_group(self, t):
        'expression : LPAREN expression RPAREN'
        t[0] = t[2]
        # t[0] = Node("exp_group", [t[2]], None)

    def p_expression_number(self, t):
        'expression : NUMBER'
        '''expression : NUMBER 
                    | XINP'''
        t[0] = t[1]
        # t[0] = Node("number", None, t[1])

    # def p_factor_num(p):
    #  'factor : NUMBER'
    #  p[0] = p[1]

    # def p_expression_name(self, t):
    #     'expression : NAME'
    #     try:
    #         t[0] = self.names[t[1]]
    #     except LookupError:
    #         print("Undefined name '%s'" % t[1])
    #         t[0] = 0

    def p_error(self, t):
        print("Syntax error at '%s'" % t.value)

def parse(string):
    return Parser().parse(string, lexer=Lexer())

if __name__ == "__main__":
    parse("4+7/x[0]")