import random
import pandas as pd
from numpy import exp
import numpy as np
from tqdm import tqdm
import pyro
import pyro.distributions as dist
from utilities import compute_real_ctr

# vars which describes the user
EXTRA_VARS = 4

def RandomSelection(dataset):
    
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
    
    # initialize
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    
    for i in tqdm(range(N_users)):
        # randomly select an ad
        ad = random.randrange(N_ads)
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click      
        
    return ad_list, total_reward, ctr, real_ctr, regret_list


def ExploreFirst(dataset, eps = 0.3):
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    N_explore = int(N_users* eps)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    average_reward = [0] * N_ads
    
    for i in tqdm(range(N_users)):
        if i <= N_explore:
            ad = random.randrange(N_ads)
        else:
            ad = average_reward.index(max(average_reward))
            
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
                average_reward[j] = ctr_j
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
        
    return ad_list, total_reward, ctr, real_ctr, regret_list
    

def EpsilonGreedy(dataset, eps = 0.1):
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    average_reward = [0] * N_ads
    
    for i in tqdm(range(N_users)):
        # toss a coin
        p = random.random()
        
        # if result smaller than eps -> exploration
        if p <= eps:
            # randomly select an ad
            ad = random.randrange(N_ads)
        # else -> exploitation
        else:
            # choose the ad with the highest average reward
            if max(average_reward) != 0:
                ad = average_reward.index(max(average_reward))
            else:
                ad = random.randrange(N_ads)
                
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
                average_reward[j] = ctr_j
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
        
    return ad_list, total_reward, ctr, real_ctr, regret_list


def BoltzmannExploration(dataset, tau = 1):
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    average_reward = [0] * N_ads
    probs = [0] * N_ads
    
    for i in tqdm(range(N_users)):
        # generate probabilities
        #probs = (exp(average_reward / ([tau] * N_ads))) / (exp(average_reward / ([tau] * N_ads)).sum())
        
        probs = exp([r/tau for r in average_reward])
        sum_probs = probs.sum()
        probs = [p/sum_probs for p in probs]
                 
        probs = np.cumsum(probs)
        
        # toss a coin
        p = random.uniform(0, probs[-1])
        
        # select the correct ad using dichotomic search
        ad = 0
        while p >= probs[ad]:
            ad += 1
                
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
                average_reward[j] = ctr_j
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
        
    
        
    return ad_list, total_reward, ctr, real_ctr, regret_list
    
    
def UCB(dataset):
    eps = 10e-8
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    average_reward = [0] * N_ads
    
    for i in tqdm(range(N_users)):
        U = [average_reward[j] + np.sqrt(2 * np.log(i + eps) / (impressions[j] + eps)) for j in range(N_ads)]
        
        ad = U.index(max(U))
                
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
                average_reward[j] = ctr_j
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
        
    return ad_list, total_reward, ctr, real_ctr, regret_list


def ThompsonSampling(dataset, a = 1, b = 1):
    
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    average_reward = [0] * N_ads
    
    # initialize alphas and betas for the prior
    alpha = [a] * N_ads
    beta = [b] * N_ads
    
    theta = [0] * N_ads
    
    for i in tqdm(range(N_users)):
        
        for k in range(N_ads):
            theta[k] = pyro.sample('theta', 
                                   dist.Beta(alpha[k], beta[k])).item()
            
        ad = theta.index(max(theta))
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update prior
        alpha[ad], beta[ad] = alpha[ad] + did_click, beta[ad] + 1 - did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
                average_reward[j] = ctr_j
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
        
    return ad_list, total_reward, ctr, real_ctr, regret_list, alpha, beta
    
    
    
def LinUcbDisjoint(dataset, ads, alpha = 1):
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    
    # initializa A and b
    As = [np.identity(EXTRA_VARS) for _ in range(N_ads)]
    Bs = [np.zeros(EXTRA_VARS) for _ in range(N_ads)]
    theta = [(np.linalg.inv(As[i])).dot(Bs[i]) for i in range(N_ads)]
    p = [0] * N_ads
    x = [0] * N_ads
    
    # context dataset
    contexts = dataset[["age", "sex", "occupation", "interested_in"]]
    ads_context = [np.asarray(ads[i].arr()) for i in range(N_ads)]
    
    for i in tqdm(range(N_users)):
        # observe context
        for j in range(N_ads):
            x[j] = ads_context[j]
            #x[j] = np.concatenate((ads_context[j], contexts.values[i,]))
        
        theta = [(np.linalg.inv(As[i])).dot(Bs[i]) for i in range(N_ads)]
        for j in range(N_ads):
            A_inv = np.linalg.inv(As[j])
            theta[j] = A_inv.dot(Bs[j])
            
            t1 = (np.transpose(theta[j])).dot(x[j])
            t2 = ((np.transpose(x[j])).dot(A_inv)).dot(x[j])
            
            p[j] = t1 + alpha*np.sqrt(t2)
            
        # take best ad
        maxs = np.where(p == np.amax(p))
        ad = random.choice(maxs[0])
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update A and b
        As[ad] = As[ad] + x[ad].dot(np.transpose(x[ad]))
        Bs[ad] = Bs[ad] + did_click*x[ad]
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
    
    return ad_list, total_reward, ctr, real_ctr, regret_list



def LinUCB(dataset, ads, alpha = 1):
    # obtain number of ads and number of users
    N_ads = len(dataset.columns) - EXTRA_VARS
    N_users = len(dataset)
    
    # evaluate real ctr
    real_ctr = compute_real_ctr(dataset, N_users, N_ads)
        
    # init
    regret = 0
    total_reward = 0
    regret_list = []
    ctr = {i:[] for i in range(N_ads)}
    ad_list = []
    max_ctr = max(real_ctr.values())
    
    impressions = [0] * N_ads
    clicks = [0] * N_ads
    
    # initializa A and b
    d = 2
    A = np.identity(EXTRA_VARS * d)
    b = np.zeros(EXTRA_VARS * d)
    
    p = [0] * N_ads
    x = [0] * N_ads
    
    # context dataset
    contexts = dataset[["age", "sex", "occupation", "interested_in"]]
    ads_context = [np.asarray(ads[i].arr()) for i in range(N_ads)]
    
    for i in tqdm(range(N_users)):
        # observe context
        for j in range(N_ads):
            if d == 1:
                x[j] = ads_context[j]
            else:
                x[j] = np.concatenate((ads_context[j], contexts.values[i,]))
        
        A_inv = np.linalg.inv(A)
        theta = A_inv.dot(b)
        
        for j in range(N_ads):            
            t1 = (np.transpose(theta)).dot(x[j])
            t2 = ((np.transpose(x[j])).dot(A_inv)).dot(x[j])
            p[j] = t1 + alpha*np.sqrt(t2)
            
        # take best ad
        maxs = np.where(p == np.amax(p))
        ad = random.choice(maxs[0])
        ad_list.append(ad)
        
        # incremente impressions for that ad
        impressions[ad] += 1
        
        # check if user clicked
        did_click = dataset.values[i,ad]
        
        if did_click:
            clicks[ad] += did_click
            
        # update A and b
        A = A + x[ad].dot(x[ad])
        b = b + x[ad]*did_click
            
        # update ctr of the ads
        for j in range(N_ads):
            if impressions[j] != 0:
                ctr_j = clicks[j] / impressions[j]
            else:
                ctr_j = 0
                
            ctr[j].append(ctr_j)
            
        # final updates
        regret += max_ctr - real_ctr[ad]
        regret_list.append(regret)
        total_reward += did_click 
    
    return ad_list, total_reward, ctr, real_ctr, regret_list
        
        
        
        
            
    
    
    
    
    
    