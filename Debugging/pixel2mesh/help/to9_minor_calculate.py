import numpy as np 

def convex_polygon(n, k=3):
    if k!=3: 
        return "Shit this function cannot do this man"
    else:
        return 3*(n-2), 2*(n-2) # The first is the edges, second is faces



if __name__ == "__main__":
    n = np.array([156, 618, 2466])
    edge, face = convex_polygon(n)
    pass
