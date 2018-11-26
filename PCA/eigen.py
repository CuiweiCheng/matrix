import numpy as np
import sys
import time
import pandas as pd


def _eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def _estimate_spectrum(matrix, tolerance):
    """
    input 1 : matrix to be used to calculate spectrum (eigenvalues and eigenvectors)
    input 2 : tolerance, a positive number less than 1
    output 1 : evalist, result of eigenvalues
    output 2 : vlist, result of eigenvectors
    The function estimates spectrum using the power method 
    and it terminates when the current eigenvalue < tolerance * the top eigenvalue
    """
    
    if not (tolerance>0 and tolerance<1):
        print('The tolerance input is out of range. Please input a tolerance which is a positive number less than 1.')
        return ([],[])
    else: 
        n = matrix.shape[0]
        evalist = []
        vlist = []
        evamax = 0
        T = 1000 # number of iterations to compute w in the power method
        print('The program is still running.')
         
        start = time.clock()
        for k in range(n):  # compute the k th eigenvalue and eigenvector 
            v = np.zeros(n)
            w = np.zeros(n)
            for j in range(n):
                v[j] = np.random.uniform(0,1)
           
            for t in range(T):
                w = matrix.dot(v) 
                normw = (np.inner(w,w))
                normw = np.sqrt(normw)     
                v = w/normw
                
            eva = _eigenvalue(matrix,v)
    
            if eva < evamax * tolerance: # check the tolerance
                break
            evalist.append(eva)
            vlist.append(v)
            evamax = max(evamax,eva)
            
            # change the matrix to compute the next eigenvalue and eigenvector
            vmat=np.mat(v)
            temp = np.dot(vmat.T,vmat)
            matrix = matrix - eva * temp
            matrix = matrix.getA()
        
        end = time.clock()
        print('It takes',end-start,'seconds when T=',T)
        
        return (evalist, vlist)


def calculate_return_rate(asset_pool_pd):
    # convert price into return
    asset_pool_pd = asset_pool_pd / asset_pool_pd.shift(axis=1) - 1
    asset_pool_pd = asset_pool_pd.drop(columns = 0)
    return asset_pool_pd
        
def calculate_cov(asset_pool_pd):
    # compute Cov matrix
    start = time.clock()
    M, N = asset_pool_pd.shape
    f = lambda x: x - x.mean()
    temp1 = asset_pool_pd.apply(f, axis = 1)
    temp2 = temp1.T
    cov_matrix = np.dot(temp1, temp2) / (N-1)
    end = time.clock()
    print ("It takes", end-start, "seconds to compute Cov matrix using our own algorithm.")
    return cov_matrix

def calculate_eigens(asset_pool_pd,tolerance):
    asset_pool_pd=calculate_return_rate(asset_pool_pd)
    cov_matrix=calculate_cov(asset_pool_pd)
    [evalist, vlist] = _estimate_spectrum(cov_matrix, tolerance)
    return evalist,vlist