# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys
from robustwritelp import writelp
from mysolver import lpsolver
    
def detect_robust_arb(arbdat,sigdat,lpfile,solutionfile):
    
    """
        Detect the existence of a TYPE A Arbitrage.
        Using daily price data of numsec number of securities
        
        Parameters
        ----------
        arbdat : file name
            The first array to pass to the rolling {human_readable}.
        rhs : array-like
            The second array to pass to the rolling {human_readable}.
        window : int
            Size of the rolling window in terms of the periodicity of the data.
        out : array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.
        **kwargs
            Forwarded to :func:`~empyrical.{name}`.
        Returns
        -------
        rolling_{name} : array-like
            The rolling {human_readable}.
        """
        
    datafile= open(arbdat, 'r')
    
    lines = datafile.readlines();
    datafile.close()
    
    firstline = lines[0].split()
    
    numsec = int(firstline[1])
    numscen = int(firstline[3])
    r = float(firstline[5])
    print ("\n")
    print ("number of securities:", numsec,"number of scenarios", numscen,"r",r)
    print ("\n")
    
    #allocate prices as one-dim array
    total = (1 + numsec)*(1 + numscen)
    print ("total allocation:",  total)
    p = [0]*total
    k = 0
    # line k+1 has scenario k (0 = today)
    while k <= numscen:
        thisline = lines[k + 1].split()
        # should check that the line contains numsec + 1 words
    
        p[k*(1 + numsec)] = 1 + r*(k != 0) # handles the price of cash
        j = 1
        while j <= numsec:
            value = float(thisline[j])
            p[k*(1 + numsec) + j] = value
            # print ">>", "k",k, "j",j, k*(1 + numsec) + j
            j += 1
        k += 1
    
    sig=[0]*(numsec+1)*(1 + numscen)
    
    
    datafile= open(sigdat, 'r')
    #try:
    #    datafile = open(sys.argv[2], 'r') # opens the data file
    #except IOError:
    #    sys.exit("Cannot open file %s\n" % sys.argv[2])
    lines = datafile.readlines();
    datafile.close()
    k=1
    while k <= numscen:
        thisline = lines[k-1].split()
        j = 0
        while j <= numsec:
            value = float(thisline[j+1])
            sig[(k)*(1 + numsec) + j] = value
            # print ">>", "k",k, "j",j, k*(1 + numsec) + j
            j += 1
        k += 1
    
    lpwritecode = writelp(lpfile, p, sig , numsec, numscen)
    
    print ("wrote LP to file", lpfile, "with code", lpwritecode)
    
    #now solve lp 
    
    lpsolvecode = lpsolver(lpfile, solutionfile)
    
    print ("solved LP at", lpfile,"with code", lpsolvecode)