import numpy as np
from scipy import stats
from scipy.stats import poisson, norm
import multiprocessing as mp

# AMS metrics
def ams(s,b,sys,ams_type):
    breg=1.    
    if ams_type=='naive1':
        if b==0:
            b += breg
        return s/np.sqrt(b+(sys*b)**2)

    if ams_type=='naive2':
        return s/np.sqrt(s+b)

    if ams_type=='naive3':
        return np.sqrt(s+b)-np.sqrt(s)

    if ams_type=='llr':
        if b==0:
            b += breg
        return np.sqrt(2.0)*np.sqrt( (s+b)*np.log(1.+s/b)-s )

    if ams_type=='LiMa':
        if b==0. and sys!=0.:
            aux=np.sqrt(2*s*(1.+1./sys))
        else:
            if b==0:
                b += breg
            if sys==0.:
                aux=np.sqrt(2.0)*np.sqrt( (s+b)*np.log(1.+s/b)-s )
            else:
                aux=np.sqrt(2.0)*np.sqrt((s+b)*np.log((1.+1./(b*sys**2))*(s+b)/(s+b+1./sys**2))+(1./sys)**2*np.log((1.+b*sys**2)*(1/sys**2)/(s+b+1./sys**2)))
        return aux  


# Log-Likelihood Ratio statistic computation
# log-likelihhod ratio function
def llr(s,b,d):
    return np.nan_to_num(s-d*np.log(1.+s/b))

def binning(X,y,w,classes,nbins,L):
    # data size
    n_data, ncol = X.shape

    # creating binned histograms of chosen variables
    # Let's normalize variables to [0,1]
    dx=np.max(X,axis=0)-np.min(X,axis=0)
    Xn = (X-np.min(X,axis=0))/dx

    # first binning, we need two classes: signal and total background
    s=np.array([Xn[i] for i in range(n_data) if y[i]==classes[0]])
    b=np.array([Xn[i] for i in range(n_data) if y[i]!=classes[0]])
    ws=np.sum(np.array([w[i] for i in range(n_data) if y[i]==classes[0]]))*L #number of signal events
    wb=np.sum(np.array([w[i] for i in range(n_data) if y[i]!=classes[0]]))*L #number of background events

    # unnormalized PDFs of the kinematic variables
    hs=[[] for i in range(ncol)]
    hs=np.array([np.histogram(s[:,i], bins=nbins, range=(0,1))[0] for i in range(ncol)])
    hb=[[] for i in range(ncol)]
    hb=np.array([np.histogram(b[:,i], bins=nbins, range=(0,1))[0] for i in range(ncol)])


    # second binning, no empty bins allowed
    fb=[[] for i in range(ncol)]
    fs=[[] for i in range(ncol)]

    for k in range(ncol):
        for i in range(nbins):
            if hb[k][i]!=0:
                fb[k].append(hb[k][i])                   
                fs[k].append(hs[k][i])
            else:    
                fb[k].append(sum(hb[k][i-1:]))
                fs[k].append(sum(hs[k][i-1:]))
                if i==0:
                    fb[k].append(1.)
                    fs[k].append(sum(hs[k][i-1:]))
                break

    # weighted PDFs
    fb = np.array([np.asarray(fb[i],dtype=float) for i in range(ncol)])
    fs = np.array([np.asarray(fs[i],dtype=float) for i in range(ncol)])
    fb *= wb/sum(fb[0])
    fs *= ws/sum(fs[0])

    return fs, fb

#def sample():
#    # Re-seed the random number generator
#    random_state = np.random.seed()
#    sp = poisson.rvs(p, size=nbins, random_state=random_state)
#    return sp

def _parallel_poisson(ntreads,niter,nbins,p0,p1,isb,nev,eps):
    pool = mp.Pool(ntreads)

    future_res = [pool.apply_async(sample) for _ in range(niter)]
    res = [f.get() for f in future_res]

    print future_res

    return np.asarray(res)

# Generating LLR PDF with pseudo-experiments
def pdf_llr(npseudos,ncol,fs,fb,sys):
    
    vllr_null=[]
    vllr_altr=[]
    
    nevB=sum(fb[0])
    eps_b=sys*nevB
    breg=1.
    if nevB==0:
        nevB += breg
    
    for n in range(npseudos):
        # incorporating simple systematics in background normalization
        fbsys = fb*max(0.,norm.rvs(nevB,eps_b)/nevB)

        aux0=0
        aux1=0
        for i in range(ncol):

            nbins=len(fb[i])
            
            d0=poisson.rvs(fbsys[i], size=nbins)
            d1=poisson.rvs(fs[i]+fbsys[i], size=nbins)

            aux0=aux0+np.sum(llr(fs[i],fb[i],d0))
            aux1=aux1+np.sum(llr(fs[i],fb[i],d1))
        
            vllr_null=np.append(vllr_null,aux0)
            vllr_altr=np.append(vllr_altr,aux1)
        
    return vllr_null, vllr_altr

# Simplified approach for the significance calculation from LLR
def ams_llr(X,y,w,classes,nbins,npseudos,sys,L):
    
    n_data, ncol = X.shape
    fs, fb = binning(X,y,w,classes,nbins,L)
    pdf_llr_null, pdf_llr_altr = pdf_llr(npseudos,ncol,fs,fb,sys)
    
    ms=np.mean(pdf_llr_altr) 
    mb=np.mean(pdf_llr_null) 
    sts=np.std(pdf_llr_altr)
    stb=np.std(pdf_llr_null)
    sigmas=abs((ms-mb)/stb)
    return ms, mb, sts, stb, sigmas, pdf_llr_null, pdf_llr_altr
    

# BOOLEAN TYPE OF CUT
def comp_bool(x,y,tipo,x0):
    if tipo=='gt':
        if x>=y:
            return True
        else:
            return False
    else:
        if tipo=='lt':
            if x<=y:
                return True
            else:
                return False
        else:
            if tipo=='sym_in':
                if np.abs(x-x0)<=y: #mass inside the window
                    return True
                else:
                    return False
            else:
                if np.abs(x-x0)>=y: #mass inside the window
                    return True
                else:
                    return False

# TYPES OF CUTS
def type_cut(X,y,classes,ncutvars,cutvars,skew_param):
    cut_type={}

    ndata=len(y)
    nclasses=len(classes)
    for Ix in range(ncutvars):

        dist_med =np.zeros(nclasses)
        dist_skew=np.zeros(nclasses) 
        count=0
        for classe in classes: 
        #        print stats.skew(dist)
            dist_med[count] =np.median(np.array([X[i,Ix] for i in range(ndata) if y[i]==classe]))
            dist_skew[count]=stats.skew(np.array([X[i,Ix] for i in range(ndata) if y[i]==classe]))  #stats.skew(dist)
            count += 1

        print 'Classes Median = ', dist_med
        if max(abs(dist_skew))<skew_param:
            print 'Classes Skew  = ', dist_skew

        if abs(dist_skew[0])>skew_param:
            if dist_med[0]<max(dist_med[1:]):
                tipo='lt'
                xc=0
                #print cutvars[Ix], tipo
            else:
                tipo='gt'
                xc=0
                #print cutvars[Ix], tipo
        else:
            tipo='sym_in'
            xc=np.median(np.array([X[i,Ix] for i in range(ndata) if y[i]==classes[0]]))
            #print cutvars[Ix], tipo, xc
        # fill the vector
        cut_type[Ix]=[tipo,xc]

    return cut_type


# PASSCUTS FUNCTION
def passcuts(v,c,tipo):
    res_bool=True
    for i in range(len(c)):
        res_bool = res_bool and comp_bool(v[i],c[i],tipo[i][0],tipo[i][1])
    return res_bool 

# CONSTRAINTS
def constraint(nS,nB,cS,cB,tipo):
    if tipo=='min_events':
        return nS>cS and nB>cB 
    else:
        if tipo=='sob':
            if nB>0:
                return nS/nB > cS
            else:
                return True


# a dictionary merger
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z



