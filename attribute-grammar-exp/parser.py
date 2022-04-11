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
    ε = 1e-8
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
    
    def p_expression_plus(self, p):
        'expression : expression PLUS term'
        p[0] = f"{p[1]}+{p[3]}"
    
    def p_expression_minus(self, p):
        'expression : expression MINUS term'
        p[0] = f"{p[1]}-{p[3]}"
    
    def p_expression_term(self, p):
        'expression : term'
        p[0] = p[1]
    
    def p_term_times(self, p):
        'term : term TIMES factor'
        p[0] = f"{p[1]}*{p[3]}"
    
    def p_term_div(self, p):
        'term : term DIVIDE factor'
        p[0] = f"{p[1]}/({p[3]}+{self.ε})"
    
    def p_term_factor(self, p):
        'term : factor'
        p[0] = p[1]
    
    def p_factor_num(self, p):
        '''factor : NUMBER 
                | XINP'''
        p[0] = p[1]
    
    def p_factor_expr(self, p):
        'factor : LPAREN expression RPAREN'
        p[0] = p[2]
    
    # Error rule for syntax errors
    def p_error(self, p):
        print("Syntax error in input!")

def parse_(string):
    return Parser().parse(string, lexer=Lexer())

if __name__ == "__main__":
    expression = parse_("4+7/x[0]")
    print(expression)