from ply import lex
from ply.lex import Token
import re

tokens = (
    'SYMBOL',
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
    symbol = r"S_?\d+"
    @Token(symbol)
    def t_SYMBOL(self, token):
        string_value = str(token.value)
        match = re.search(r"\d+", string_value) 
        if match:
            pos = int(match[0]) 
        match = re.search(r"_", string_value)
        bit = 0 if match else 1
        token.value = (pos, bit)    
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
    data = 'S_10S8S_5S9S10S10S_9S_1S1S_1'
    lexer_.input(data)
    while True:
        tok = lexer_.token()
        if not tok: 
            break      # No more input
        print(tok)