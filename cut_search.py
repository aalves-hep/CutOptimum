def main(kwargs):


    print('--------- PARAMETERS -----------\n')
    print(kwargs)

    import csv
    import numpy as np
    from utilities import ams, ams_llr, constraint, type_cut, passcuts
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 
    import matplotlib.pyplot as plt

    from utilities import pdf_llr, merge_two_dicts, binning

    params={}
    for k,v in kwargs.items():
        params[k]=v

    #reading parameters
    input_dir     =params['input_dir']
    input_file    =params['input_file']
    output_dir    =params['output_dir']
    output_file1  =params['output_file1']
    output_file2  =params['output_file2']
    output_file3  =params['output_file3']
    isllr         =params['isLogLRatio']
    lbins         =params['num_llr_bins']
    nthreads      =params['n_threads']
    npseudos      =params['npseudos']
    cutvars       =params['cutvariables']
    mvavars       =params['mvariables']
    auto_type     =params['auto_type']
    cut_type      =params['cut_type']
    limiar_space  =params['limiar_space']
    limiar_manual =params['limiar_manual']
    esp           =params['espaco']
    n_auto_bins   =params['auto_bins']
    isconstraint  =params['isconstraint']
    constraints   =params['constraints']
    nevals        =params['evaluations']
    L             =params['Luminosity']
    sys           =params['systematics']
    ams_type      =params['ams_type']
    skew_param    =params['skew_param']
    optimize      =params['isopt'] 
    search_type   =params['search_type']
    plotter       =params['isplot']
    writer        =params['iswrite']
    
    # loading data
    # Slicing off header row and id, weight, and label columns.
    data = list(csv.reader(open(input_dir+input_file,"rb"), delimiter=','))

    # reading variables names
    names = data[0]
    variables={}
    count=0
    for name in names:
        variables[name]=count
        count += 1

    # data vector
    data = np.array([map(float, row) for row in data[1:]])
    n_raw, ncol = data.shape
    print('\n')
    print ('finish loading '+str(n_raw)+' events from csv file\n')

    # raw data
    X_raw = np.delete(data,np.s_[names.index('weight'),names.index('class')],1) # FEATURES MATRIX
    y_raw = data[:,names.index('class')].astype(int) # TARGETS
    w_raw = data[:,names.index('weight')] # WEIGHTS

    # obtaining the classes
    classes=np.arange(min(y_raw),max(y_raw)+1)
    nclasses=len(classes)

    print 'CLASSES: ', classes

    weights=np.array([np.sum(np.array([w_raw[i] for i in range(n_raw) if y_raw[i]==classe])) for classe in classes])

    wS, wB = weights[0], np.sum(weights[1:]) 
    print 'xsecs = ', weights, wS, wB,'\n'

    ###############################
    # CUT-AND-COUNT: TPE/HyperOpt #
    ###############################

    # Cut variables
    #cutvars=cutvars
    cutindex=np.sort([variables[v] for v in cutvars])
    ncutvars=len(cutindex)

    X_var=np.array([X_raw[:,i] for i in cutindex]).T
    y_var=y_raw

    # MVA variables
    #if isllr:
        #mvavars=mvavars
    mvaindex={variables[v] for v in mvavars}
    nmvavars=len(mvaindex)    
    X_mva=np.array([X_raw[:,i] for i in mvaindex]).T
            

    # SIMPLE DECISION OF THE TYPE OF CUTS
    print('-------- CUT STRATEGY ----------')
    if auto_type:
        cut_type = type_cut(X_var,y_var,classes,ncutvars,cutvars,skew_param)
    else:
        cut_type = cut_type
    print cut_type

    # CONSTRAINTS
    constraintS, constraintB, ctype = constraints

    # CUTS DICTIONARY
    cuts={}
    if not optimize:

        limiar=limiar_manual
        X = X_var
        for i in range(ncutvars):
            cuts[cutvars[i]]=limiar[i]

        print('------- MANUAL CUTS --------')
        print cuts

        # OBJECTIVE FUNCTION
        def objective(cuts):

            cut=np.array([cuts[cutvars[k]] for k in range(ncutvars)])
            data_cut=np.array([data[i] for i in range(n_raw) if passcuts(X[i],cut,cut_type)])
        
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
                    if constraint(nevS,nevB,constraintS,constraintB,ctype): #constraintS and nevB>constraintB:
                        sigmas = ams(nevS,nevB,sys,ams_type)
                        if isllr:
                            X_mva = np.array([X_cut[:,i] for i in mvaindex]).T
                            res_mva = ams_llr(X_cut,y_cut,w_cut,lbins,npseudos,sys,L)
                    else:
                        sigmas = 0.                
            else:
                sigmas = 0.
                nev = np.zeros(nclasses)
            return {'sigmas':sigmas, 'NEvents': nev, 'mva':res_mva, 'status': STATUS_OK}

        result=objective(cuts)
        significance_cut = result['sigmas']
        nev = result['NEvents']

        print '------------ RESULTS (ONLY CUTS) --------------'
        print 'Number of Events in Classes = ', nev
        print 'sys, Best CC/TPE, S, B, S/B =', sys, significance_cut , nev[0], sum(nev[1:]), nev[0]/sum(nev[1:])

        print '------------ RESULTS (CUTS+LLR) --------------'
        res_mva = result['mva']
        print 'Best CC/TPE MVA =', res_mva[4] 


    else:
        if esp=='user':

            limiar = limiar_space
            X = X_var
            for i in range(ncutvars):
                cuts[cutvars[i]]=hp.quniform(cutvars[i]+'_cut',limiar[i][0],limiar[i][1],limiar[i][2])

        else:
            if esp=='auto':  ### STILL BUGGED 01/06

                limiar = [[] for i in range(ncutvars)]
                # MinMax transformation of the features
                xmin=np.min(X_var,axis=0)
                xmax=np.max(X_var,axis=0)
                dx=np.abs(xmax-xmin)
                X = (X_var-xmin)/dx

                for i in range(ncutvars):
                    limiar[i]=[0.,1.,1./n_auto_bins]

                for i in range(ncutvars):
                    cuts[cutvars[i]]=hp.quniform(cutvars[i]+'_cut',limiar[i][0],limiar[i][1],limiar[i][2])
                

        # OBJECTIVE FUNCTION
        def objective(cuts):

            cut=np.array([cuts[cutvars[k]] for k in range(ncutvars)])
            data_cut=np.array([data[i] for i in range(n_raw) if passcuts(X[i],cut,cut_type)])
        
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
                    loss = -ams(nevS,nevB,sys,ams_type)
                    if isllr:
                        X_mva = np.array([X_cut[:,i] for i in mvaindex]).T
                        res_mva = ams_llr(X_mva,y_cut,w_cut,lbins,npseudos,sys,L)
                        loss = -res_mva[4] #loss
                else:
                    if constraint(nevS,nevB,constraintS,constraintB,ctype): #constraintS and nevB>constraintB:
                        loss = -ams(nevS,nevB,sys,ams_type)
                        if isllr:
                            X_mva = np.array([X_cut[:,i] for i in mvaindex]).T
                            res_mva = ams_llr(X_mva,y_cut,w_cut,lbins,npseudos,sys,L)
                            loss = -res_mva[4] #-loss
                    else:
                        loss = 0.                
            else:
                loss = 0.
                nev=np.zeros(nclasses)
            return {'loss':loss, 'NEvents': nev, 'status': STATUS_OK}


        print '----- HyperOpt SEARCH '+str(nevals)+' trials ------'
        trials = Trials()
        if search_type=='tpe':
            algo=tpe.suggest
        else:
            algo=random.suggest

        best = fmin(fn=objective,
                    space=cuts,
                    algo=algo,
                    max_evals=nevals,
                    trials=trials)
        
        print 'Best Cuts:'
        print best

    # best ams calculation
        cut={}
        for i in range(ncutvars):
            cut[i]=best[cutvars[i]+'_cut']

        data_best=np.array([data[i] for i in range(n_raw) if passcuts(X_var[i],cut,cut_type)])
        
        X_best = data_best[:,:-2]
        y_best = data_best[:,-1].astype(int)
        n_best = len(y_best)
        w_best = data_best[:,-2]
        
        weights=np.array([np.sum(np.array([w_best[i] for i in range(n_best) if y_best[i]==classe])) for classe in classes])
    
        nev = L*weights
        nevS, nevB = nev[0], np.sum(nev[1:])
        significance_cut = ams(nevS,nevB,sys,ams_type)
        sob = nevS/nevB

        print '------------ RESULTS (ONLY CUTS) --------------'
        print 'Number of Events in Classes = ', nev
        print 'sys, Best CC/TPE, S, B, S/B =', sys, significance_cut , nevS, nevB, sob
        
        if isllr:
            X_mva  = np.array([X_best[:,i] for i in mvaindex]).T

            n_data, ncol = X_mva.shape
            #fs, fb = binning(X_mva,y_best,w_best,lbins,L)
            res_mva = ams_llr(X_mva,y_best,w_best,lbins,10*npseudos,sys,L)
            pdf_llr_null = res_mva[5]
            pdf_llr_altr = res_mva[6]
            
            # Plotting llr PDFs
            plt.figure()
            plt.hist(pdf_llr_null, bins=50, lw=2, color='blue', histtype='step', label='H0')
            plt.hist(pdf_llr_altr, bins=50, lw=2, color='red',  histtype='step', label='H1')
            plt.title(str(10*npseudos)+' pseudoexperiments')
            plt.legend(loc='best')
            plt.savefig(output_dir+output_file3)
            plt.show()

            #print np.mean(pdf_llr_null), np.mean(pdf_llr_altr)
            #print np.abs(np.mean(pdf_llr_null)-np.mean(pdf_llr_altr))/np.std(pdf_llr_null)

            significance_mva = res_mva[4]
            print '------------ RESULTS (CUTS+LLR) --------------'
            print 'Best CC/TPE MVA =', significance_mva
        else:
            significance_mva = 0

        if writer:
            f = open(output_dir+output_file1, 'w')

            res = {'Luminosity': L,
                   'systematics': sys, 
                   'Significance(CUTS)': significance_cut,
                   'Significance(MVA)': significance_mva,
                   'AMS Type': ams_type,
                   'S/B Ratio': sob,
                   'nevS': nevS,
                   'nevB': nevB,
                   'Number of searches': nevals}

            newres = merge_two_dicts(res, best)
            f.write(str(newres))
            f.close()

        #vcuts=np.zeros(10)
        nbins=5
        eval_bin=nevals/nbins
        if plotter:
            vetor = trials.trials
            vams  = -1*np.array([trials.trials[i]['result']['loss'] for i in range(nevals)])
            
            
            # Cut-and-Count
            vcuts=np.array([np.array([vams[k] for k in range((i+1)*eval_bin)]) for i in range(nbins)])
            

            vbest=np.array([np.max(vams[:i]) for i in range(1,nevals)])
            vx=np.arange(1,nevals)

            fig = plt.figure()
            ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
            ax2 = fig.add_axes([0.15, 0.6, 0.3, 0.2]) # inset axes

            # main figure
            ax1.hist(vcuts[0], nbins, normed=False, color='blue', lw=2, label=str(eval_bin)+' trials', histtype='step')
            ax1.hist(vcuts[1], nbins, normed=False, color='red',  lw=2, label=str(2*eval_bin)+' trials', histtype='step')
            ax1.hist(vcuts[2], nbins, normed=False, color='orange',  lw=2, label=str(3*eval_bin)+' trials', histtype='step')
            ax1.hist(vcuts[3], nbins, normed=False, color='green',  lw=2, label=str(4*eval_bin)+' trials', histtype='step')
            ax1.hist(vcuts[4], nbins, normed=False, color='black',  lw=2, label=str(5*eval_bin)+' trials', histtype='step')
            #ax1.plot([2.1, 2.1], [0, 124], color='black', linestyle='--', linewidth=2)
            #plt.text(2.05, 70, 'best manual search',  rotation=90)
            ax1.set_xlabel(r'Signal Significance ($\sigma$)', fontsize=15)
            ax1.set_ylabel(r'Frequency', fontsize=15)
            #plt.ylim([0,150])
            #plt.yscale('log', nonposy='clip')
            ax1.set_title(r'GP optimized search, '+str(nevals)+' trials', fontsize=20)
            ax1.legend(loc="best")
            # insert
            ax2.plot(vx, vbest, lw=2, color='black', label=r' ')
            #ax2.plot([0, 1000], [2.1, 2.1], color='black', linestyle='--', linewidth=2)
            ax2.set_xlim([1,nevals])
            ##ax2.set_ylim([1.8,2.9])
            #ax2.text(140, 2.15, r"default cuts", fontsize=13, color="black")
            #ax2.set_xlabel(r'Trials', fontsize=12)
            #ax2.set_title(r'Significance ($\sigma$)', fontsize=12)
            ##ax2.set_ylabel(r'Max ($\sigma$)', fontsize=15)
            ax2.grid(True, lw = 0.4, ls = '-', c = '.25')
            #ax2.tight_layout()
            plt.savefig(output_dir+output_file2)
            plt.show()
