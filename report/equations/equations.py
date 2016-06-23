#!/usr/bin/python3
from sympy import *

init_printing()

q00, q01, q02, q03 = symbols('q_00 q_01 q_02 q_03')
q11, q12, q13 = symbols('q_11 q_12 q_13')
q22, q23 = symbols('q_22 q_23')
q33 = symbols('q_33')

allQ = [q00, q01, q02, q03, q11, q12, q13, q22, q23, q33]

Pix0, Pix1, Pix2, Pix3 = symbols('zR_x0, zR_x1, zR_x2, zR_x3')
Piy0, Piy1, Piy2, Piy3 = symbols('zR_y0, zR_y1, zR_y2, zR_y3')
Piz0, Piz1, Piz2, Piz3 = symbols('zR_z0, zR_z1, zR_z2, zR_z3')

Pix = Matrix([[Pix0], [Pix1], [Pix2], [Pix3]])
Piy = Matrix([[Piy0], [Piy1], [Piy2], [Piy3]])
Piz = Matrix([[Piz0], [Piz1], [Piz2], [Piz3]])

Q = Matrix([[q00, q01, q02, q03], [q01, q11, q12, q13], [q02, q12, q22, q23], [q03, q13, q23, q33]])

eq1 = (Pix.T * Q * Pix - Piy.T * Q * Piy)[0]
eq2 = (Pix.T * Q * Piy)[0]
eq3 = (Piy.T * Q * Piz)[0]
eq4 = (Piz.T * Q * Pix)[0]
eq5 = (Piz.T * Q * Piz)[0]

eq1 = rcollect(expand(eq1), *allQ)
eq2 = rcollect(expand(eq2), *allQ)
eq3 = rcollect(expand(eq3), *allQ)
eq4 = rcollect(expand(eq4), *allQ)
eq5 = rcollect(expand(eq5), *allQ)

def extract_coefficients(expr):
    def index(args):
        for i in range(len(args)):
            if args[i] in allQ:
                return i

        return 0

    D = { x.args[index(x.args)] : Mul(*(x.args[:index(x.args)] + x.args[index(x.args) + 1:])) for x in expr.args }
    return [D[q] for q in allQ]

def make_row(expr):
    return ' & '.join([latex(expr).replace('zR', 'P') for expr in extract_coefficients(expr)])

system = open("system.tex", "w+")
system.write("\\center \\begin{rotate}{270}")
system.write("\\smatheq {\n\t\\matrix {\n")
system.write("\t" + make_row(eq1) + "\\\\\n")
system.write("\t" + make_row(eq2) + "\\\\\n")
system.write("\t" + make_row(eq3) + "\\\\\n")
system.write("\t" + make_row(eq4) + "\\\\\n")
system.write("\t" + make_row(eq5) + "\\\\\n")
system.write("\t} \cdot \n")

system.write("\t\\matrix {\n")
for q in allQ:
    system.write(latex(q) + "\\\\")
system.write("\t}\n}\n\\end{rotate}")







a, b, c, d, e, f, g, h, i, j, k, l = symbols('a b c d e f g h i j k l')

A = Matrix([
    [a, b, c],
    [d, e, f],
    [g, h, i],
    [j, k, l]])

Q = A * A.T

print(Q)
print(Q[0,:])
print(Q[1,:])
print(Q[2,:])
print(Q[3,:])