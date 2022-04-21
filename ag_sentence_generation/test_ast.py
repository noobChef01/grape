import ast


text = '''
def factorial(n):
    if n == 1:
        return 1
    return n*factorial(n-1)
'''

print(ast.dump(ast.parse(text, mode='eval')))