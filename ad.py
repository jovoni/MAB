import pyro
import random

class Ad():
    
    def __init__(self):
        self.target_age = 'young'
        self.target_sex = 'male'
        self.occupation = 'student'
        self.item_sold = random.randint(1,5)
        self.appeal = pyro.sample('appeal', pyro.distributions.Uniform(0.1,0.5)).item()
        self.generate()
        
    def generate(self):
        a = pyro.sample('a', pyro.distributions.Uniform(0,1)).item()
        if a <= 1/3:
            self.target_age = 1
        elif a > 1/3 and a <= 2/3:
            self.target_age = 2
        else:
            self.target_age = 3
        
        s = pyro.sample('s', pyro.distributions.Uniform(0,1)).item()
        if s <= 0.5:
            self.target_sex = 1
        else:
            self.target_sex = 2
            
        i = pyro.sample('i', pyro.distributions.Uniform(0,1)).item()
        if i <= 0.2:
            self.occupation = 1
        elif i > 0.2 and i < 0.8:
            self.occupation = 2
        else:
            self.occupation = 3
            
    def __str__(self):
        return f"Ad: ({self.target_age}, {self.target_sex}, {self.occupation}, {self.item_sold})"
    
    def arr(self):
        return [self.target_age, self.target_sex, self.occupation, self.item_sold]