import sys
import numpy as np

def evaluate(x,a,b,c,k1,k2,f1,f2):
    """
    Evaluates the sinusoidal model
    """
    return a * np.sin(k1 * x - f1) + b * np.cos(k2 * x - f2) + c

def setup(params):
    """
    Sets up the environment and returns the arguments for running the model
    """

    modelpath = params["modelpath"]

    return ["python", modelpath] + [str(params[i]) for i in ["x","a","b","c","k1","k2","f1","f2"]]

if __name__ == "__main__":
    x,a,b,c,k1,k2,f1,f2 = sys.argv[1:]
    result = evaluate(x,a,b,c,k1,k2,f1,f2)
    print(result)
