## Small scirpt for calculating autocorrelation times and error estimates form MD data ###

#===========================================================================================================================================================================================================
# Please read & cite the following references when using this program:
#
# [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium #states.
# J. Chem. Phys. 129:124105, 2008
# http://dx.doi.org/10.1063/1.2978177
#
# [2] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
# histogram analysis method for the analysis of simulated and parallel tempering simulations.
# JCTC 3(1):26-41, 2007.
# 
# [3] J.D Chodera, A Simple Method for Automated Equilibration Detection in Molecular Simulations
# J. Chem. Theory Comput., 2016, 12 (4), pp 1799–1805 10.1021/acs.jctc.5b00784
#
# Also check out the following documentation or ref. 4 for more details:
# 
# http://pymbar.readthedocs.io/en/latest/timeseries.html
#
#
# [4] F. Grunewald, G. Rossi, A. H. de Vries, S. J. Marrink, L. Monticelli, A transferable MARTINI model of PEO, in preparation
#
#===========================================================================================================================================================================================================

import numpy as np
import argparse
import pymbar as pm
import pandas as pd

#===========================================================================================================================================================================================================
#                                                                                           Importing Data
#===========================================================================================================================================================================================================
"""
This is faster than numpy.loadtxt plus it discards the header. Adapted from:
https://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy
"""

def iter_loadtxt(filename, delimiter=' ', skiprows=0):
    def skip_header(skiprows):
        with open(filename, 'r') as infile:
             for line in infile:
                 if any([ word in '[ @, #, @TYPE ]' for word in line.split()]):              
                    skiprows = skiprows + 1
                 else:
                    return(skiprows)

    def iter_func(skiprows):
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                for item in line.replace('\n', ' ').split():
                    yield item
        iter_loadtxt.rowlength = len(line.replace('\n', ' ').split())

    skiprows = skip_header(skiprows)
    data = np.fromiter(iter_func(skiprows),dtype=float)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


#===========================================================================================================================================================================================================
#                                        Functions to determine the statistical inefficency and subsample data after Chodera and Shirts
#===========================================================================================================================================================================================================
""" We determine the number of uncorrelated samples from the statistical inefficency
    following the procedure proposed by Shirts & Chodera determine [1]. There is also 
    the option to performe some analysis of possible equilibriation times following
    the procedure outlined in [3]. 
"""

def decorr_no_eq(data):
    gIn = pm.timeseries.statisticalInefficiency(data,fast=args.fast)
    indices = pm.timeseries.subsampleCorrelatedData(data, g=gIn,fast=args.fast)
    data_uncorr = data[indices]
    return(data_uncorr, gIn)

def decorr_eq(data):
    [t0, g, Neff_max] = pm.timeseries.detectEquilibration(data,fast=args.fast) # compute indices of uncorrelated timeseries
    A_t_equil = data[t0:]
    indices = pm.timeseries.subsampleCorrelatedData(A_t_equil, g=g,fast=args.fast)
    A_n = A_t_equil[indices]
    return(t0, A_n, g)

def perform_analysis(data_set,eq,minimum,ID,store,time_step):
    try:        
        if not eq:
           data_uncorr, g = decorr_no_eq(data_set)
           t_eq=0
        else:
           t_eq, data_uncorr, g = decorr_eq(data_set)
           t_eq = t_eq * time_step
 
        n_uncorr = len(data_uncorr)
    
        if n_uncorr > minimum:
           tau = ((g-1) / 2.0) * time_step # this is the estimate of the autocorrelation time 
           mean = np.average(data_uncorr)
           std_error = np.std(data_uncorr,ddof=1)/np.sqrt(len(data_uncorr))
           results_table = pd.DataFrame([[mean, std_error,n_uncorr,g,tau,t_eq]],columns=['mean','std-error','n-uncorr.','g','tau','t-eq'],index=[ID])    
        else:
           print("\n+++++++++++++++++++++++++ WARNING ++++++++++++++++++++++")
           print("Only ",n_uncorr," samples found in data set ",ID ,".")
           print("This is lower than the minimum threshold of", args.min)
           print("samples. The simulation should be prologned!")
           print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
           tau = len(data_set) * time_step  
           mean = np.average(data_set)
           std_error = 'NaN'
           results_table = pd.DataFrame([[mean, std_error,n_uncorr,g,tau,t_eq]],columns=['mean','std-error','n-uncorr.','g','tau','t-eq'],index=[ID])
        if store:
           np.savetxt('decorrelated_data'+str(ID)+'.dat', data_uncorr)

    except pm.utils.ParameterError:
          print("\n+++++++++++++++++++++++++ WARNING ++++++++++++++++++++++")
          print("The statistical inefficency for data set", ID,"could")
          print("not be computed! Most likely your data set is zero or")
          print("constant. The data will not be further treated.")
          print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    return(results_table)
#==========================================================================================================================================================================================================
#                                                                                         Main
#===========================================================================================================================================================================================================

print()
f = open('stat_ref.txt','r')
pretext = f.read()
print(pretext)
f.close()

parser = argparse.ArgumentParser(description='Use timeseries analysis to compute error and autocorrelation time \n of data sets from MD simulations.\n')
parser.add_argument('-f'       , dest='infile'      , metavar='NameData'   , type=str,help='name of the data-file\n')
parser.add_argument('-o'       , dest='outfile'     , type=str ,nargs='*'  ,help='name of output-file, if -append is set multiple data-sets \n can be appended to multiple output-files\n')
parser.add_argument('-id'      , dest='id'          , type=str ,nargs='*'  ,help='names for the data-sets in file\n')
parser.add_argument('-append'  , action='store_true', help='append a existing csv file\n')
parser.add_argument('-min'     , dest='min'         , type=int , default=100,help='threshold for printing a warning message.\n')
parser.add_argument('-eq'      , action='store_true', help='perfrom equilibriation analysis.\n')
parser.add_argument('-fast'    , action='store_true', help='use fast flag to increase speed of computing statistical inefficency\n')
parser.add_argument('-print'   , dest='store'       , action='store_true' , help='save uncorrelated data for each data set.\n')
parser.add_argument('-check'   , action='store_true', help='perform analysis multiple times each time including a larger fraction of the data. \n')
parser.add_argument('-bins'    , dest='bins',type=int,help='number of bins for check analysis \n',default=50)
args = parser.parse_args()

def __main__():
    data = iter_loadtxt(args.infile)
    time_step = abs(data[0,0] - data[1,0])
    results = {}

    n_data_sets = len(data[1,:]) - 1
    output = pd.DataFrame(columns=['mean','std-error','n-uncorr.','g','tau','t-eq'])


    if args.id == None: 
       indices = pd.Series([ str(i) for i in np.arange(0,n_data_sets)])
    else:
       if len(args.id) == n_data_sets - 1:
          indices = args.id
       else:
          print("+++++++++++++++ Error: more data sets than labels ++++++++++++++++++") 
          exit()
 
    file_count=0
    for ID, data_set in zip(indices, data[:,1:].T):
        data_set = data_set.reshape(-1)

        if args.check:
           bins = args.bins
           bin_sizes = [ round(len(data_set)/bins) * n for n in np.arange(1,bins+1,1)]
           percent = [ (size/len(data_set))*100 for size in bin_sizes  ]
           binned_data = [ data_set[0:i] for i in bin_sizes ]
           results = [ perform_analysis(subset,args.eq,args.min,ID,args.store,time_step) for subset in binned_data] 
           means = np.array([ df.get_value(ID, 'mean', takeable=False) for df in results])
           taus = np.array([ df.get_value(ID, 'tau', takeable=False) for df in results])
           np.savetxt('tau_as_f_of_time'+ID+'.dat',np.transpose([ percent,taus]))
           errs = np.array([ df.get_value(ID, 'std-error', takeable=False) for df in results])
           np.savetxt('mean_as_f_of_time'+ID+'.dat',np.transpose([ percent,means,errs]))
           results_table = results[-1]      
        else:
          results_table = perform_analysis(data_set,args.eq,args.min,ID,args.store,time_step)

        if args.append:
           with open(args.outfile[file_count], 'a') as f:
                results_table.to_csv(f, header=False)  
           file_count = file_count + 1  
        else:
             output = output.append(results_table)

    if not args.append:
       with open(args.outfile[0],'w') as f:
            output.to_csv(f, sep=',')
       print('+++++++++++++++++++++++++ RESULTS ++++++++++++++++++++++++++++')
       print(output)
       print('\n')
    return(None)      

__main__() 
