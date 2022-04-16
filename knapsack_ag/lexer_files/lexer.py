from ply import lex
from ply.lex import Token

tokens = (
    'ZERO',
    'ONE'
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

    # A regular expression rule with some action code
    zero = r'0'
    @Token(zero)
    def t_ZERO(self, token):
        token.value = (int(token.value), token.lexpos)    
        return token

    one = r'1'
    @Token(one)
    def t_ONE(self, token):
        token.value = (int(token.value), token.lexpos)    
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
    data = '010101010'
    lexer_.input(data)
    while True:
        tok = lexer_.token()
        if not tok: 
            break      # No more input
        print(tok)