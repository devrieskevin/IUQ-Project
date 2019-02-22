import sys
import numpy as np

def evaluate(x,a,c,k,f):
    """
    Evaluates the cosine model
    """
    return a * np.cos(k * x - f) + c, 0.0

def evaluate_dict(params):
    """
    Evaluates the cosine model using a dictionary
    """
    
    x = params["x"]
    a = params["a"]
    c = params["c"]
    k = params["k"]
    f = params["f"]
    
    return a * np.cos(k * x - f) + c, 0.0

def setup(modelpath, params):
    """
    Sets up the environment and returns the arguments for running the model
    """

    return ["python", modelpath] + [str(params[i]) for i in ["x","a","c","k","f"]]

def read_output(path):
    with open("%s/stdout.log" % (path),'r') as infile:
        output = infile.read()
        
    return float(output), 0.0

def generate(n_samples,scale=1.0):
    params = a, c, k, f = np.random.uniform(-5, 5, size=4)
    x = np.linspace(0, 2 * np.pi, n_samples)
    
    y,_ = evaluate(x,a,c,k,f)
    eps = np.random.normal(0.0, scale, size=n_samples)
    
    data = y + eps
    err = 2.0 * np.abs(eps)
    
    return x, data, err, params
    
    
if __name__ == "__main__":
    x,a,c,k,f = [float(sys.argv[n]) for n in range(1,6)]
    result,_ = evaluate(x,a,c,k,f)
    print(result)
