from sympy import Symbol, solve

def solver(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')

    eq1 = a*x1**5 + b*x1**4 + c*x1**3 + d*x1**2 + e*x1 + f - y1
    eq2 = a*x2**5 + b*x2**4 + c*x2**3 + d*x2**2 + e*x2 + f - y2
    eq3 = a*x3**5 + b*x3**4 + c*x3**3 + d*x3**2 + e*x3 + f - y3
    eq4 = a*x4**5 + b*x4**4 + c*x4**3 + d*x4**2 + e*x4 + f - y4
    eq5 = a*x5**5 + b*x5**4 + c*x5**3 + d*x5**2 + e*x5 + f - y5
    eq6 = a*x6**5 + b*x6**4 + c*x6**3 + d*x6**2 + e*x6 + f - y6
    
    return solve([eq1, eq2, eq3, eq4, eq5, eq6], a, b, c, d, e, f)

print(solve(1,2, 2,4 ,7,8, 15, 13, ))