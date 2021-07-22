import pyro
import random

class User():
    
    def __init__(self):
        self.age = 'young'
        self.sex = 'male'
        self.occupation = 'student'
        self.interested_in = 1
        self.generate()
        
    def generate(self):
        a = pyro.sample('a', pyro.distributions.Uniform(18,80)).item()
        if a <= 30:
            self.age = 1
        elif a > 30 and a <= 60:
            self.age = 2
        else:
            self.age = 3
        
        
        s = pyro.sample('s', pyro.distributions.Uniform(0,1)).item()
        if s < 0.5:
            self.sex = 1
        else:
            self.sex = 2
            
        i = pyro.sample('i', pyro.distributions.Uniform(0,1)).item()
        if i <= 0.15:
            self.occupation = 1
        elif i > 0.15 and i < 0.9:
            self.occupation = 2
        else:
            self.occupation = 2
            
        p = pyro.sample('p', pyro.distributions.Uniform(0,25)).item()
        if p <= 1:
            self.interested_in = 1
        elif p <= 4:
            self.interested_in = 2
        elif p <= 9:
            self.interested_in = 3
        elif p <= 16:
            self.interested_in = 4
        else:
            self.interested_in = 5
        
            
    def __str__(self):
        return f"User: ({self.age}, {self.sex}, {self.occupation})"