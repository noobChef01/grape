from ply import lex
from ply.lex import Token

tokens = (
    # 'NAME','NUMBER',
    'NUMBER'
    # 'PLUS','MINUS','TIMES','DIVIDE','EQUALS',
    'PLUS','MINUS','TIMES','DIVIDE',
    'LPAREN','RPAREN', 'XINP'
    )

## Normally PLY works at the module level. Here it is encapsulated as a class. 
## Thus the strange construction of this class in the new method allows
## PLY to do its magic.
class Lexer(object):

    def __new__(cls, **kwargs):
        self = super(Lexer, cls).__new__(cls, **kwargs)
        self.lexer = lex.lex(object=self, **kwargs)
        return self.lexer

    tokens = tokens

    t_PLUS    = r'\+'
    t_MINUS   = r'-'
    t_TIMES   = r'\*'
    t_DIVIDE  = r'/'
    t_EQUALS  = r'='
    t_LPAREN  = r'\('
    t_RPAREN  = r'\)'
    t_NAME    = r'[a-zA-Z_][a-zA-Z0-9_]*'

    
    # A regular expression rule with some action code
    x_inp = r"x\[0\]"
    @Token(x_inp)
    def t_XINP(self, token):
        token.value = str(token.value)
        return token
        
    number = r'\d+'
    @Token(number)
    def t_NUMBER(self, token):
        token.value = int(token.value)    
        return token
    
    # Define a rule so we can track line numbers
    newline = r'\n+'
    @Token(newline)
    def t_newline(self, token):
        token.lexer.lineno += len(token.value)
    
    # A string containing ignored characters (spaces and tabs)
    t_ignore  = ' \t'
 
    # Error handling rule
    def t_error(self, token):
        print("Illegal character '%s'" % token.value[0])
        token.lexer.skip(1)

if __name__ == "__main__":
    lexer_ = Lexer()
    data = '''
            3 + 4 * 10
            + -20 *2*x[0]
            '''
    lexer_.input(data)
    while True:
        tok = lexer_.token()
        if not tok: 
            break      # No more input
        print(tok)