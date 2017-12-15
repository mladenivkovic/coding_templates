#!/usr/bin/python3


#===============================
# examples for sympy
#===============================

import sympy

a,b = sympy.symbols('a b')


print()
print("Expand")
print(sympy.expand((a+b)**3))

print()
print("Factorize")
print(sympy.factor(a**2-b**2))


print()
print("Differentiate")
x = sympy.symbols('x')
print(sympy.diff(sympy.sin(x), x))



print()
print("Integrate")
print(sympy.integrate(x**2,(x,a,b)))


