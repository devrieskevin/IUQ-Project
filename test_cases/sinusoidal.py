import sys
import numpy as np

def evaluate(x,a,b,c,k1,k2,f1,f2):
    """
    Evaluates the sinusoidal model
    """
    return a * np.sin(k1 * x - f1) + b * np.cos(k2 * x - f2) + c

def setup(modelpath, params):
    """
    Sets up the environment and returns the arguments for running the model
    """

    return ["python", modelpath] + [str(params[i]) for i in ["x","a","b","c","k1","k2","f1","f2"]]

def read_output(path):
    with open("%s/stdout.log" % (path),'r') as infile:
        output = infile.read()
        
    return float(output), 0.0

def generate(n_samples,scale=1.0):
    params = a, b, c, k1, k2, f1, f2 = np.random.uniform(-1, 1, size=7)
    
    x = np.linspace(0, 2 * np.pi, n_samples)
    y = evaluate(x, a, b, c, k1, k2, f1, f2)
    eps = np.random.normal(0.0, scale, size=n_samples)
    
    data = y + eps
    err = np.abs(eps)
    
    return x, data, err, params
    
    
if __name__ == "__main__":
    x,a,b,c,k1,k2,f1,f2 = [float(sys.argv[n]) for n in range(1,9)]
    
    result = evaluate(x,a,b,c,k1,k2,f1,f2)
    print(result)
