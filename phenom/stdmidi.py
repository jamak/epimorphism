def m0(f):
    return f

def m0_e(f):
    return 0.8 * f + 0.2

def m1(f):
    return 4.0 * f

def m1_e(f):
    return 3.8 * f + 0.2

def m2(f):
    return 4.0 * f - 2.0

def m3(f):
    return 5.0 * f

def m4(f):
    return 2.0 * 3.14159 * f

def m0_inv(f):
    return f

def m0_e_inv(f):
    return (f - 0.2) / 0.8

def m1_inv(f):
    return f / 4.0

def m1_e_inv(f):
    return (f - 0.2) / 3.8

def m2_inv(f):
    return (f + 2.0) / 4.0

def m3_inv(f):
    return f / 5.0

def m4_inv(f):
    return f / (2.0 * 3.14159)
