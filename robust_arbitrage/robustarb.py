# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys
from robustwritelp import writelp
from mysolver import lpsolver

#if len(sys.argv) != 4:
#    sys.exit("usage: robustarb.py datafilename sigmafilename lpfilename\n")
#
##now open and read data file
#try:
#    datafile = open(sys.argv[1], 'r') # opens the data file
#except IOError:
#    sys.exit("Cannot open file %s\n" % sys.argv[1])
    
def detect_robust_arb(arbdat,sigdat,lpfile,solutionfile):
    datafile= open(arbdat, 'r')
    
    lines = datafile.readlines();
    datafile.close()
    
    #print lines[0]
    firstline = lines[0].split()
    #print "first line is", firstline
    
    numsec = int(firstline[1])
    numscen = int(firstline[3])
    r = float(firstline[5])
    print "\n"
    print "number of securities:", numsec,"number of scenarios", numscen,"r",r
    print "\n"
    
    #allocate prices as one-dim array
    total = (1 + numsec)*(1 + numscen)
    print "total allocation:",  total
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
    
    print "wrote LP to file", lpfile, "with code", lpwritecode
    
    #now solve lp 
    
    lpsolvecode = lpsolver(lpfile, solutionfile)
    
    print "solved LP at", lpfile,"with code", lpsolvecode