import pyro
from numpy import exp
import numpy as np
import random

class Policy():

    def __init__(self, extra_vars = 4, n_ads = 5):
        #self.weights = [[pyro.sample('w', pyro.distributions.Normal(0,1)).item() for i in range(extra_vars)] for j in range(n_ads)]
        self.weights = [[random.random() for i in range(extra_vars)] for j in range(n_ads)]

    def propose(self, context, n_ads):
        probs = np.array([np.exp(np.dot(context, self.weights[i])) for i in range(n_ads)])
        print(type(probs))
        probs = probs / probs.sum()
        print(probs)
        
        return np.argmax(probs)

    def __str__(self):
        return f'Policy_weights = {self.weights}'
        

if __name__ == "__main__":

    context = [1,3,2,5]
    
    p = Policy(4, 5)

    print(p)

    c = p.propose(context, 5)

    print(c)