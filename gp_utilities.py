import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import c
from gp_algorithms import EI, PI, TS, UCB, GP_UCB

def compute_mean_and_variance(X, y, Xtest, kernel, noise_var=0.02):
    N = len(X)
    n = len(Xtest)
    
    K = kernel(X, X)
    L = np.linalg.cholesky(K + noise_var*np.eye(N))

    # compute the mean at our test points.
    Lk = np.linalg.solve(L, kernel(X, Xtest))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))

    # compute the variance at our test points.
    K_ = kernel(Xtest, Xtest)
    s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
    s = np.sqrt(s2)
    
    return mu, s


def plot_GP(X, y, Xtest, mu, s, f, noise_var, xmin, xmax, ymin, ymax):
    plt.figure(1)
    plt.clf()
    plt.plot(X, y, color=c.nordred, marker = 'p', linestyle = 'None',ms=10)
    plt.plot(Xtest, f(Xtest), color = c.brightblue, linestyle = 'dashed')
    plt.gca().fill_between(Xtest.flat, mu-1*s, mu+1*s, color=c.nordwhite)
    plt.plot(Xtest, mu, color=c.nordred, linestyle='dashed', lw=2)
    plt.title('Mean predictions plus st.deviations')
    plt.axis([xmin, xmax, ymin, ymax])
    
    
def plot_GP_and_AF(X,y,Xtest,mu,s,af_prob,f,noise_var,xmin,xmax,ymin,ymax,it,save_plot=False,method='None'):
    fig, axs = plt.subplots(2,1, sharex=True)
    #fig.title(f"Iteration = {it}")
    
    idx = best_idx(Xtest,af_prob)
    
    axs[0].plot(X, y, c=c.nordred, marker='p', linestyle='None', ms=8)
    axs[0].plot(Xtest, f(Xtest), c=c.brightblue, linestyle='dashed')
    axs[0].fill_between(Xtest.flat, mu-1*s, mu+1*s, color=c.nordwhite)
    axs[0].plot(Xtest, mu, c = c.nordred, linestyle='dashed', lw=2)
    axs[0].axvline(Xtest[idx], linestyle='dashdot', linewidth=1, c=c.nordblue)  
    axs[0].set(title=f"t = {it}")  #ylabel="Objective function"
    axs[0].axis([xmin, xmax, ymin, ymax])
    
    
    axs[1].plot(Xtest,af_prob, c=c.nordgreen)
    axs[1].set_xlim([xmin,xmax])
    #axs[1].set_ylim([0,1])
    if it == 1:
        axs[1].set(ylabel=f"{method}")
    axs[1].axvline(Xtest[idx], linestyle='dashdot', linewidth=1, c=c.nordblue)    
    axs[1].plot(Xtest[idx], af_prob[idx], marker='p', c=c.nordred, linestyle='None', ms=8)
    
    if save_plot:
        plt.savefig(f"images/{method}{it}.png", bbox_inches='tight')
    
    
def best_idx(Xtest,probs):
    return np.argmax(probs)


def next_x(Xtest, probs):
    return Xtest[best_idx(Xtest,probs)]


def update_GP(X,y,Xtest,f,noise,af,kernel,xmin,xmax,ymin,ymax,it,eps=0.1,k=1,delta=0.5,save=False,method='None'):
    # compute mean and variance
    mu, s = compute_mean_and_variance(X,y,Xtest,kernel)
    
    # compute acquisition function values
    if af == EI or af == PI:
        af_values = af(y,mu,s,eps=eps)
    elif af == UCB:
        af_values = af(mu,s,k=k)
    elif af == GP_UCB:
        af_values = af(mu,s,xmin,xmax,t=it,delta=delta)
    elif af == TS:
        af_values = af(X, y, mu,Xtest,kernel,noise)
    else:
        raise ValueError(f"Acquisition function given {af} not known")
    
    # plot
    plot_GP_and_AF(X,y,Xtest,mu,s,af_values,f,noise,xmin,xmax,ymin,ymax,it,save_plot=save,method=method)
    
    # find next value of x
    new_x = next_x(Xtest, af_values)
    
    # update X and y with new values
    X = np.append(X,new_x).reshape(-1,1)
    y = np.append(y,f(new_x) + noise*np.random.randn(1))
    
    return X,y

def sample_and_plot_posteriors(nsamples,X,y,Xtest,kernel,noise,xmin,xmax,ymin,ymax,save=False):    
    n = len(Xtest)
    N = len(X)
    
    K = kernel(X,X)
    L = np.linalg.cholesky(K + noise*np.eye(N))
    
    Lk = np.linalg.solve(L, kernel(X,Xtest))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    
    K_ = kernel(Xtest, Xtest)
    s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
    s = np.sqrt(s2)
    
    L_ = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1,1) + np.dot(L_, np.random.normal(size=(n,nsamples)))
    
    plt.figure(1)
    plt.clf()
    plt.plot(X, y, color=c.nordred, marker = 'p', linestyle = 'None',ms=10)
    plt.plot(Xtest, f_post)
    plt.gca().fill_between(Xtest.flat, mu-1.5*s, mu+1.5*s, color=c.nordwhite)
    plt.plot(Xtest, mu, color=c.nordred, linestyle='dashed', lw=2)
    plt.title(f'{nsamples} samples from the GP posterior')
    plt.axis([xmin, xmax, ymin, ymax])
    if save:
        plt.savefig('images/post_sample.png', bbox_inches='tight')
    plt.show()
    
    
def plot_afs(Xtest,af_values,xmin,xmax,par_list,method):
    n = len(par_list)
    
    if method == 'PI' or method == 'EI':
        par = '\u03B5'
    elif method == 'UCB':
        par = 'k'
    elif method == GP_UCB:
        par = '\u03B4'
    
    for i in range(n):
        plt.plot(Xtest, af_values[i], label=f'{par} = {par_list[i]}')
        
    plt.legend()
    plt.title(f"Acquisition function {method}")
    plt.xlabel("X")
    plt.ylabel("\u03B1")
    plt.show()
    
    
def compare_af_values(X,y,Xtest,af,kernel,xmin,xmax,par_list,method='None'):
    mu,s = compute_mean_and_variance(X,y,Xtest,kernel)
    
    if af == EI or af == PI:
        af_values = [af(y,mu,s,eps=i) for i in par_list]
    elif af == UCB:
        af_values = [af(mu,s,k=i) for i in par_list]
    elif af == GP_UCB:
        af_values = [af(mu,s,xmin,xmax,t=it,delta=i) for i in par_list]
    else:
        raise ValueError(f"Acceptable acquisition functions are PI,EI,UCB and GP-UCB")
        
    plot_afs(Xtest,af_values,xmin,xmax,par_list,method=method) 
    
    
    
    
    
    
    
   