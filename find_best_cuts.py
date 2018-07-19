"""
========================================================
OptCut : A Python Package for Cut-and-Count Optimization
========================================================
This is a beta version. 03/07/2018
author: Alexandre Alves

OptCut uses a Gaussian Process algorithm to search 
for rectangular cut thresholds which maximize a significance metrics

"""

print(__doc__)

from cut_search import main
import time
from datetime import date
today=date.today()
today=today.timetuple()
uc='_'

# Input yout sigopt_card.dat
f = open('sigopt_card.dat', 'r')
params = eval(f.read())
f.close()


localtime=time.time()
main(params)
localtime=time.time()-localtime
print 'Running time: '+str(localtime/60)+' minutes'
