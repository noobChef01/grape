<E>   ::= <E> + <T> {E.val = E.val + T.val} | <T> {E.val = T.val}
<T>  ::= <T> * <F> {T.val = T.val * F.val} | <F> {T.val = F.val}
<F>  ::= num {F.val = num}


<e>   ::= <e>/<e>|x[1]|<c><c>.<c><c>
<c>  ::= 0 | 1

<lettersequence> ::= <asequence> <bsequence> <csequence>
condition :
Size (<asequence>) = Size (<bsequence>) = Size (<csequence>)
<asequence> ::= a
Size (<asequence>) ← 1
| <asequence2> a
Size (<asequence>) ← Size (<asequence2>) + 1
<bsequence> ::= b
Size (<bsequence>) ← 1
| <bsequence2> b
Size (<bsequence>) ← Size (<bsequence2>) + 1
<csequence> ::= c
Size (<csequence>) ← 1
| <csequence2> c
Size (<csequence>) ←Size (<csequence2>) + 1