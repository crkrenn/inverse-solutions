def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def boundary(fn, x):
    target = fn(x)
    def boundary_fn(xb):
        return (target - fn(xb))**2
    return boundary_fn
