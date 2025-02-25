import rcca
import numpy as np

def run_cca(X, Y, Z, numCC=2,reg=0.):
    # Create a RCCA object
    numCC = int(numCC)
    cca = rcca.CCA(kernelcca = False, reg = reg, numCC = numCC)
    
    # Train the model on three datasets
    cca.train([X, Y, Z])

    # Return the canonical coefficients for each dataset
    return cca.ws
  
  
