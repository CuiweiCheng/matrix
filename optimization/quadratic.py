import numpy as np
from numpy import linalg as LA
import pandas as pd
import sys
    
def _compute_fx(x, mu, cov, lambda0):

    xt_cov = np.dot(x,cov)
    xt_cov_x = np.dot(xt_cov,x) # 1*1
    mut_x = np.dot(mu,x) # 1*1
    fx = lambda0 * xt_cov_x - mut_x
    
    return fx

def _compute_gx(x, mu, cov, lambda0):
    cov_covt = cov + cov.T
    cov_covt_x = np.dot(cov_covt,x) # 4*1
    gx = lambda0 * cov_covt_x - mu
    return gx

def _find_m(x,l,u,d):
    # input the descent of F(x) which is denoted by "descent" 
    descent = np.array(d)
    # initialize the minimium value
    minimize = descent.dot(descent)
    # initialize final value of y
    final_y=[0,0,0,0]
    # read data that includes lower bound, upper bound and x. Sort the descent variable in descending order
    input_sort = pd.DataFrame()
    input_sort['descent'] = d
    input_sort['x'] = x
    input_sort['l'] = l
    input_sort['u'] = u
    input_sort.sort_values(by = "descent", ascending = False, inplace = True)
    input_sort['y'] = final_y
    

    # implement algorithm mentioned in lecture
    
    #when j <m ,yj =lj-xj
    #when j >m ,yj =uj-xj
    #for ym, we can compute it by substracting the sum of remaining of y from 0.
    
    for m in range(4):
        input_sort['y'] = np.zeros(len(x))
        input_sort.iloc[0:m,4] = input_sort.iloc[0:m,2] - input_sort.iloc[0:m,1]
        input_sort.iloc[m+1:,4] = input_sort.iloc[m+1:,3] - input_sort.iloc[m+1:,1]
        input_sort.iloc[m,4] = 0 - sum(input_sort.iloc[:,4])
        input_sort['x_new'] = input_sort['x'] + input_sort['y']

        
        tempxy = np.around(input_sort['x_new'],decimals = 5)
        
        # we need to make sure that new value satisfy the constraint.
        if sum(tempxy <= input_sort['u']) == 4 and sum(tempxy >= input_sort['l']) == 4:
            #   find the minimum of product of descent and y
            input_sort['gy'] = input_sort['descent'] * input_sort['y']
            product = sum(input_sort['gy'])

            if product<minimize:

                minimize = product
                output_sort = input_sort.copy()
                output_sort.sort_index(inplace = True)                
                final_y = output_sort['y'].values

            else:
                continue
        else:
            continue
        
    return final_y

def _find_s(x, y, u, l,mu,covariance,lambdaval):
    s_min=0    

    #s takes value from 0 to 1
    s = np.arange(0, 1, 0.0001)
    F0 = float('inf')
    result = [F0]
    for i in range(len(s)):
        # assign new value to x0 so as to compute F(x0)
        x0 = x + np.dot(s[i], y)

        # we need to check whether the requirement is meet or not
        if sum(x0 <= u) == 4 and sum(x0 >= l) == 4 and sum(x0) == 1:
            #conpute the value of F(x0)
            F1 = _compute_fx(x0,mu,covariance, lambdaval)
            result.append(F1)
            # in every iteration , we keep F0 as minimum and compare it with new F(x)
            if F1 < F0:
                F0 = F1
                s_min = s[i]
                x_new = x0
            else:
                continue
        else:
            continue
    return (x_new, s_min, min(result))

def quadratic_opt(n,lam,matrix,covariance):
    
    """
    Compute the weight of n portfolios according to Markowitz Model.
    Utilize quadratic optimization to decide the optimal weight of the assets.
    
    Parameters
    ----------
    n : Integer
        n denotes the number of assets which compose the portfolio
    lam : any real number (float)
        lam denote the lambda in the Markowitz Model, which reflects the Risk Attitude
        of the investor.
        If lam is larger, the more Risk-appetite the investor is. 
        If lam is smaller, the more Risk-averse the investor is.
    matrix: pd.DataFrame
        consists of n rows, each row i has three attribute:
            lower, the lower bound of the weight of asset i
            upper, the upper bound of the weight of asset i
            mu, the return rate of asset i (usually use daily data)
    covariance: pd.DataFrame
        the covariance matrix of rate of return of the n assets (usually use daily data)
        
    Returns
    -------
    x_new : n-dimension vector (list), which denote the optimal weight of the n assets
        Assuming the sum of each dimension in x_new equals to 1
    F_new : the objective function in Markowitz Model, denoting the minimum value of the objective function 
    """
    
    df1 = matrix
    df2 = covariance
    lambdaval=lam
    lower = np.array(df1['lower'])
    upper = np.array(df1['upper'])
    mu = np.array(df1['mu'])
    print("Lambda is :", lambdaval)
    print('\n')
    print("Bounds of assets and expexted return of assets:\n", df1)        
    print('\n')
    print("Covariance matrix: \n", df2)
    
    ''' -------------------------- Find First Feasible x -------------------------- '''
    print('\n','\n','\n','\n','\n')
    print('lower','\n',lower,'\n','upper','\n',upper,'\n',)
    x = np.zeros(n)
    if sum(lower) < 1:
        for i in range(len(lower)):
            x[i] = lower[i]
            
        for i in range(4):
            temp = sum(upper[0:i+1])+sum(lower[i+1:4])
            if temp <= 1:
                x[i] = upper[i]
            else:
                mark = i
                break
      
        while sum(x) < 1:
            x[mark] += 0.001 # increment step
    elif sum(lower) == 1:
        return 'Feasible at lower bound'
    else:
        return 'No Feasible solution'
    
    x = np.around(x, decimals = 5)
    fx = _compute_fx(x, mu, covariance.values, lambdaval)
    gx = _compute_gx(x, mu, covariance.values, lambdaval)
    u=upper #upper bound
    l=lower#lower bound
    
    descent = gx
    y = _find_m(x, l, u, descent)
    s_min = 0
    F1 = _compute_fx(x, mu, covariance.values, lambdaval)
    [x_new, s, F_new] = _find_s(x, y, u, l,mu,covariance.values,lambdaval)

    
    Fx_list = []
    
    ''' Using Iteration to minimize the F-value '''
    print('Now iterating the algorithm to find out the minimum...')
    for i in range(1000):
        Fx_list.append(F_new)
        gx = _compute_gx(x_new, mu, covariance.values, lambdaval)
        descent = gx
        y = _find_m(x_new, l, u, descent)
        [x_new, s, F_new] = _find_s(x_new, y, u, l,mu,covariance.values,lambdaval)
        print('x=',x_new,'F=',F_new)
        # Calculate the difference between two F-value
        temp = abs(Fx_list[-1] - F_new)/Fx_list[-1]
        if  temp < 0.000001: 
            break
    '''tolerance (the difference between two F-value < tolrance,then we stop iteration)'''
    ''' Print the optimal portolio arrangement and the min of F-value '''
    print('x=',x_new,'F=',F_new)
    return [x_new,F_new]