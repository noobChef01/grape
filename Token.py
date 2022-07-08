T_NT = 'NT'
T_T = 'T'


class Token(object):

    def __init__(self, type, name, meta=None) -> None:
        self.type = type
        self.name = name
        self.meta = meta

    def __str__(self) -> str:
        '''
        String representation of class instance.

        Examples: 
            Token(NT, '<expr>', None)
            Token(T, 'X1', {"value": 10, "weight": 50})
        '''
        return f"Token({self.type}, {self.name}, {self.meta})"

    def __repr__(self) -> str:
        return self.__str__()
