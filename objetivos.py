import numpy as np
from utilities import ams, ams_llr, constraint, passcuts
from hyperopt import STATUS_OK

# OBJECTIVE FUNCTION
def objective_manual(data,X,n_raw,
                     cuts,cutvars,cut_type,classes,
                     L,sys,ams_type,
                     isconstraint,constraints,
                     isllr,mvaindex,lbins,npseudos,
                     isopt):

    res_mva = []
    ncutvars = len(cutvars)
    nclasses = len(classes)

    cut=np.array([cuts[cutvars[k]] for k in range(ncutvars)])
    data_cut=np.array([data[i] for i in range(n_raw) if passcuts(X[i],cut,cut_type)])

    # CONSTRAINTS
    if isconstraint:
        constraintS, constraintB, ctype = constraints
        
    if len(data_cut)!=0:
        
        X_cut = data_cut[:,:-2]
        y_cut = data_cut[:,-1].astype(int)
        n_cut = len(y_cut)
        w_cut = data_cut[:,-2]
        
        weights=np.array([np.sum(np.array([w_cut[i] for i in range(n_cut) if y_cut[i]==classe])) for classe in classes])
                
        # number of events of the classes
        nev = L*weights
        nevS, nevB = nev[0], np.sum(nev[1:])
                
        if not isconstraint:
            sigmas = ams(nevS,nevB,sys,ams_type)
            if isllr:
                X_mva = np.array([X_cut[:,i] for i in mvaindex]).T
                res_mva = ams_llr(X_mva,y_cut,w_cut,lbins,npseudos,sys,L)
            else:
                res_mva = []
        else:
            if constraint(nevS,nevB,constraintS,constraintB,ctype): #constraintS and nevB>constraintB:
                sigmas = ams(nevS,nevB,sys,ams_type)
                if isllr:
                    X_mva = np.array([X_cut[:,i] for i in mvaindex]).T
                    res_mva = ams_llr(X_cut,y_cut,w_cut,lbins,npseudos,sys,L)
                else:
                    res_mva = []
            else:
                sigmas = 0.                
    else:
        sigmas = 0.
        nev = np.zeros(nclasses)

    if isopt:
        if not isllr:
            sigmas = -sigmas
        else:
            sigmas = -res_mva[4]

    return {'loss':sigmas, 'NEvents': nev, 'mva':res_mva, 'Xcut': X_cut}

