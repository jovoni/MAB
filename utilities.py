from user import User
from ad import Ad
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import c

# vars which describes the user
EXTRA_VARS = 4

def print_ads(ads):
    for a in ads:
        print(a)

def compute_real_ctr(dataset, N_users, N_ads):
    real_ctr = {}
    for i in range(0, N_ads):
        real_ctr[i] = (dataset[dataset.columns[i]].sum() / N_users)
    return real_ctr

def generate_clicks(user, ads, p = 0.1, age = 0.03, occupation = 0.02, sex = 0.04, item = 0.4):
    n = len(ads)
    clicks = [0] * n
    
    for i in range(n):
        ad = ads[i]
        k = p + ad.appeal
        if user.age == ad.target_age:
            k = k + age
        else:
            k = k - age/2
        if user.occupation == ad.occupation:
            k = k + occupation
        else:
            k = k + occupation/2
        if user.sex == ad.target_sex:
            k = k + sex
        else:
            k = k + sex/2
        if user.interested_in == ad.item_sold:
            k = k + item
            
        if k > random.random():
            clicks[i] = 1
            
    return clicks

def generate_dataset(N_user = 1000, N_ads = 10, file_name = "dataset.csv"):
    ads = [Ad() for _ in range(N_ads)]
    
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
    
        header = [f'ad{i}' for i in range(N_ads)] + ['age', 'sex', 'occupation', "interested_in"]
        writer.writerow(header)
    
        for i in range(N_user):
            u = User()
            r = [u.age, u.sex, u.occupation, u.interested_in]
            r = generate_clicks(u, ads) + r
            writer.writerow(r)
            
    return ads
            
def print_best_ad(df, N_users, N_ads):
    # evaluate real ctr
    real_ctr = compute_real_ctr(df, N_users, N_ads)
    #printe them
    for i in range(0,len(df.columns)-EXTRA_VARS):
        print(f'Ad #{i} CTR = {real_ctr[i]:.2f}')
        
    print(f'The Ad with the best CTR is: {max(real_ctr, key=real_ctr.get)}')
            

def print_info(N_ads, ad_list, total_reward):
    ad_shown = set(ad_list)
    count_series = pd.Series(ad_list).value_counts(normalize=True)
    for i in range(N_ads):
        if i in ad_shown:
            print(f'Ad #{i} has been shown {count_series[i]*100:.2f} % of the time.')
        else:
            print(f'Ad #{i} has been shown {00.00} % of the time.')
            
    
    print('\nTotal Reward (Number of Clicks):', total_reward)
    
    
def print_ctrs(N_users, N_ads, ctr):
    x = np.arange (0, N_users, 1)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for i in range(N_ads):
        ax1.scatter(x=x, y=ctr[i], s=2, label=f'Ad #{i}')
        
    plt.legend()
    plt.show()
    
def print_regret(N_users, regret_list, color = c.nordmagenta):
    x = np.arange (0, N_users, 1)
    
    plt.scatter(x=x, y=regret_list, s=2, c = color,label = "Regret")
    plt.legend()
    plt.show  
    
    
def print_all(N_users, N_ads, ad_list, total_reward, ctr, real_ctr, regret_list):
    print_info(N_ads=N_ads, ad_list=ad_list, total_reward=total_reward)
    print_ctrs(N_users=N_users, N_ads=N_ads, ctr=ctr)
    print_regret(N_users=N_users, regret_list=regret_list)
    