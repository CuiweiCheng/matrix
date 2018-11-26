# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys

def writelp(lpfilename, prices, sigma, numsec, numscen):

    try:
        lpfile = open(lpfilename, 'w') # opens the file
    except IOError:
        print("Cannot open LP file %s for writing\n" % lpfilename)
        return 1
    
    print "now writing LP to file", lpfilename
    lpfile.write("Minimize ")
    j = 0
    while j <= numsec:
        if prices[j] >= 0:
            lpfile.write("+ ")
        lpfile.write(str(prices[j]) + " x" + str(j)+" ")
        # str converts argument into string;  " +" concatenates strings
        j += 1
    lpfile.write("\nSubject to\n")

    k = 1
    while k <= numscen:
        # write constraint for scenario k
        
        lpfile.write("scen_" + str(k) +"17: ")
        j = 0
        while j <= numsec:
            index = k*(1 + numsec) + j
            if prices[index]-sigma[index] >= 0:
                lpfile.write("+ ")
            lpfile.write(str(prices[index]-sigma[index]) + " u" + str(k) + str(j)+" ")
            if -prices[index]-sigma[index] >= 0:
                lpfile.write("+ ")
            lpfile.write(str(-prices[index]-sigma[index]) + " v" + str(k) + str(j)+" ")
            j += 1
        lpfile.write(" >= 0\n")
#       above write(17)
        j=0
        while j <= numsec:
            lpfile.write("scen_" + str(k) +"18: ")
            lpfile.write(" u" + str(k) + str(j)+" "+" -v" + str(k) + str(j)+" "+" -x" + str(j)+" ")
            lpfile.write(" = 0\n")
            j+=1
#       above write(18)
        
        k += 1



    lpfile.write("Bounds\n")
    j = 0
    while j <= numsec:
        lpfile.write("-1 <= x" + str(j) + " <= 1\n")
        j += 1
    k = 1
    j = 0
    while k <= numscen:
        j=0
        while j <= numsec:
            lpfile.write(" u" + str(k) + str(j) + " >= 0\n")
            lpfile.write(" v" + str(k) + str(j) + " >= 0\n")
            j+=1
        k+=1
        
    lpfile.write("End\n")

    print "closing lp file"
    lpfile.close()
      
    return 0

