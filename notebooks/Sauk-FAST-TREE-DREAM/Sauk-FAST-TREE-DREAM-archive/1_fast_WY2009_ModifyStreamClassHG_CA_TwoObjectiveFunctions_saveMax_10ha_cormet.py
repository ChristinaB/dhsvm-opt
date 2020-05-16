"""
Script to run FAST(Fourier amplitude sensitivity testing).
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from pysqlite3 import dbapi2 as sqlite3
import os
import subprocess
from distutils.dir_util import remove_tree
import shutil
import time
import spotpy
import pandas as pd
from collections import OrderedDict
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename='fast.log', level=logging.INFO, filemode="w")
plt.switch_backend('agg')

class fast_run_setup(object):
    def __init__(self, parallel='seq'):        
        self.params = [spotpy.parameter.Uniform('exponential_decrease_62',low=0.5, high=3,  optguess=1.5),
                       spotpy.parameter.Uniform('lateral_conductivity_62',low=0.0002, high=0.0015,  optguess=0.0008),
                       spotpy.parameter.Uniform('exponent',low=-.20, high=-.15,  optguess=-.17)
                       ]



        self.evals1 = pd.read_csv(VALIDATION1_CSV)['value'].values
        self.evals2 = pd.read_csv(VALIDATION2_CSV)['value'].values
        self.parallel = parallel

    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    #setting up simulation for location:12189500 with predefined params and writing to config file 
    def simulation(self, x):
        pid = str(os.getpid())
        logging.info("Initiating Copy for Process %d", format(pid))
        child_dir = "./" + DIR_PREFIX + pid
        shutil.copytree(".", child_dir, ignore=shutil.ignore_patterns(DIR_PREFIX + "*", DB_NAME + "*"))
        logging.info("Copy for Process completed %d", format(pid))
        logging.info("Forking into %s",  child_dir)
        os.chdir(child_dir)

        #write DREAM parameter input to config file. 

        #soil parameters
        #loam and roccky colluvium a function of sandy loam based on values in Rawls et al., 1982
        #sandy loam, soil type 62, sat K and exponential decrease determined by script
        change_setting(CONFIG_FILE, "Exponential Decrease 62", str(round(x[0],5)))
        change_setting(CONFIG_FILE, "Lateral Conductivity 62", str(round(x[1],5)))
        change_setting(CONFIG_FILE, "Maximum Infiltration 62", str(round(x[1]*2,5))) #assume equalt to 2*saturated hydraulic conductivity
#        change_setting(CONFIG_FILE, "Porosity 62"," ".join([str(round(x[2],5)),str(round(x[2]-.05,5)),str(round(x[2]-.1,5))]))
        change_setting(CONFIG_FILE, "Vertical Conductivity 62"," ".join([str(round(x[1],5)),str(round(x[1],5)),str(round(x[1],5))]))
        
        #loam - sat K and exponential decrease 5 times less than sandy loam, porosity equal to sandy loam layer 1 (to account for high depth)
        change_setting(CONFIG_FILE, "Exponential Decrease 61", str(round(x[0]/5,5)))
        change_setting(CONFIG_FILE, "Lateral Conductivity 61", str(round(x[1]/5,5)))
        change_setting(CONFIG_FILE, "Maximum Infiltration 61", str(round(x[1]/5*2,5)))
 #       change_setting(CONFIG_FILE, "Porosity 61"," ".join([str(round(x[2],5)),str(round(x[2],5)),str(round(x[2],5))]))
        change_setting(CONFIG_FILE, "Vertical Conductivity 61"," ".join([str(round(x[1]/5,5)),str(round(x[1]/5,5)),str(round(x[1]/5,5))]))
        
        #rocky colluvium -treat as coarse sand - sat K and exponential decrease are 2 to 3 times greater than sandy loam, porosity .1 to .15 less than sandy loam
        change_setting(CONFIG_FILE, "Exponential Decrease 65", str(round(x[0]*2,5)))
        change_setting(CONFIG_FILE, "Lateral Conductivity 65", str(round(x[1]*3,5)))
        change_setting(CONFIG_FILE, "Maximum Infiltration 65", str(round(x[1]*3,5)))
#        change_setting(CONFIG_FILE, "Porosity 65"," ".join([str(round(x[2],5)),str(round(x[2]-.1,5)),str(round(x[2]-.15,5))]))
        change_setting(CONFIG_FILE, "Vertical Conductivity 65"," ".join([str(round(x[1]*3,5)),str(round(x[1]*3,5)),str(round(x[1]*3,5))]))

        write_streamclass(stream_geometry,x[2],stream_class_file)

        #run DHSVM with modified parameters in config file
        subprocess.call(DHSVM_CMD, shell=True, stdout=False, stderr=False)
        simulations=[]
        #read streamflow data from DHSVM output file
        with open(STREAMFLOW_ONLY, 'r') as file_output:
            header_name = file_output.readlines()[0].split(' ')

        with open(STREAMFLOW_ONLY) as inf:
            next(inf)
            date_q = []
            q_12189500 = []
            q_12186000 = []
            for line in inf:
                parts = line.split()
                if len(parts) > 1:
                    date_q.append(parts[0])
                    q_12189500.append(float(parts[10])/(3600*1)) #12189500 is Sauk at Sauk
                    q_12186000.append(float(parts[374])/(3600*1)) #1218600 us Sauk above Whitechuck
        os.chdir("..")
        logging.info("Removing copied directory %s", str(child_dir))
        remove_tree(child_dir)
        logging.info("Removed directory %s",  str(child_dir))

        simulation_streamflow = pd.DataFrame({'x[0]':date_q, 'x[180]':q_12189500,'x[5676]':q_12186000})
        simulation_streamflow.columns = [header_name[0], header_name[10], header_name[374]] 
        simulations1 = simulation_streamflow['180'].values
        simulations2 = simulation_streamflow['5676'].values
        simulations = (simulations1, simulations2) #tuple of simulation time series
        return simulations
	
    
    def evaluation(self): #tuple of evaluation time series
        evaluations = (self.evals1.tolist(),self.evals2.tolist())
        return evaluations
    
#    def objectivefunction(self, simulation, evaluation, params=None):
#        assert len(evaluation) == len(simulation), "Evaluation and simulation file are of different length, quitting now"
#        try:
#            model_fit = spotpy.objectivefunctions.nashsutcliffe(evaluation,simulation)
#            logging.info('Nashsutcliffe: %s', str(model_fit))
#        except Exception as e:
#            logging.info('Exception occured: %s', str(e))
#        return model_fit
		
    def objectivefunction(self, simulation, evaluation, params=None):
        '''
        test objective function that uses two flow observations
        #THIS NEEDS TO BE DEBUGGED
        '''
        evaluation1 = evaluation[0]
        evaluation2 = evaluation[1]
        simulation1 = simulation[0]
        simulation2 = simulation[1]
		
        assert len(evaluation1) == len(simulation1), "Evaluation 1 and simulation 1 file are of different length, quitting now"
        assert len(evaluation2) == len(simulation2), "Evaluation 2 and simulation 2 file are of different length, quitting now"
        try:
            model_fit1 = spotpy.objectivefunctions.nashsutcliffe(evaluation1,simulation1)
            model_fit2 = spotpy.objectivefunctions.nashsutcliffe(evaluation2,simulation2)
            model_fit = model_fit1+model_fit2
            logging.info('Nashsutcliffe: %s', str(model_fit))
			
            ###
            #write additional metrics of the simulation in this section
            
            max1 = max(simulation1)
            max2 = max(simulation2)

            with open('sim1.csv','a') as fd:
	            fd.write(str(max1)+',')
            with open('sim2.csv','a') as fd:
	            fd.write(str(max2)+',')
            ###
			
        except Exception as e:
            logging.info('Exception occured: %s', str(e))
        return model_fit
		
		

#Function to write to DHSVM config file simulated inputs from DREAM.
def change_setting(config_file, setting_name, new_value, occurrence_loc='g'):
    sed_cmd = "sed -i 's:{setting_name} = .*:{setting_name} = {new_value}:{occurrence_loc}' {config_file}"
    sed_cmd = sed_cmd.format(setting_name = setting_name, new_value = new_value
                             , config_file = config_file
                             , occurrence_loc = occurrence_loc)
    return subprocess.call(sed_cmd, shell=True)
	
	

#Function to write to stream class file based on initial set of values based on field conditions
#works
def write_streamclass(stream_geometry,cc,sname):
    #edit to read in new stream geometry
                          
    ca = stream_geometry['CAavg'].iloc[::-1].reset_index(drop=True)
    slp = [0]#Stream_Geometry['slp [m/m]']#.iloc[::-1]
    wdt = stream_geometry['W'].iloc[::-1].reset_index(drop=True)
    dep = stream_geometry['D'].iloc[::-1].reset_index(drop=True)
    mn =  stream_geometry['nb_a'].iloc[::-1].reset_index(drop=True)
#    cc1 = cc[0]
#    cc2 = cc[1]
    mn = 1.6642*ca**(cc) #function of contributing area
    
    
    dat = []
    c=1
    for c1,i in enumerate(slp):
        for c2,j in enumerate(ca):
            #print(c1)
            r2 = c #channel class
            r3 = dep.loc[c2] #depth
            r4 = wdt.loc[c2] #width
            r5 = wdt.loc[c2] #width
            r6 = mn.loc[c2] #mannings n
            
            #create list of stream parameters for each class            
            dat.append(OrderedDict({'#ID':r2,'W':r4,'D':r3,'n':r6}))
     
            c +=1

    #write stream.class.dat file
    df = pd.DataFrame(dat)
    df.to_csv('network/' + sname, sep = ' ',index=False)
    #change to
    #df.to_csv('/network/stream.class.calibration.dat',sep = ' ',index=False)
    #return subprocess.call(df, shell=True) # can delete return 


def plot_results(results, threshold=0.2):
    names = []
    values = []
    no_names = []
    no_values = []
    index = []
    no_index = []

    parnames = spotpy.analyser.get_parameternames(results)
    sensitivity_data = spotpy.analyser.get_sensitivity_of_fast(results,M=4)
    sensitivity_data = list(sensitivity_data.values())[1]
    
    with open('sensitivity_data_run_pnnlNov2006.txt', 'w') as f:
        f.writelines([str(pname) + ", " + str(s) + "\n" for (pname, s) in zip(parnames, sensitivity_data)])

    for j in range(len(sensitivity_data)):
        if sensitivity_data[j] > threshold:
            names.append(parnames[j])
            values.append(sensitivity_data[j])
            index.append(j)
        else:
            no_names.append(parnames[j])
            no_values.append(sensitivity_data[j])
            no_index.append(j)

    fig = plt.figure(figsize=(16,6))
    ax = plt.subplot(1,1,1)
    ax.bar(index, values, align='center')
    ax.bar(no_index, no_values, color='orange', label = 'Insensitive parameter')

    ax.plot(np.arange(-1,len(parnames)+1,1),[threshold]*(len(parnames)+2),'r--')
    plt.xticks(list(range(len(sensitivity_data))), parnames,rotation = 15)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    fig.savefig('FAST_sensitivity_run_pnnlNov2006.png',dpi=300)

def writeblankcsv(name):
    bd = pd.DataFrame([])
    bd.to_csv(name)
    

if __name__ == '__main__':

    #Files and their input paths
    NUM_OF_ITERATIONS = 592
    DB_NAME = 'fast_dhsvm_pnnlWY2009_592_cormet'
    CONFIG_FILE = 'Input.sauk.dynG.pnnl.WY2009_fixedPor_CA_AL_LowP_1hr_CorMet'
    STREAMFLOW_ONLY = 'output/PNNLWRF/Streamflow.Only'
    stream_geometry = pd.read_csv('StreamGeometry.csv')
    stream_class_file = 'stream.class.calibration_CA.dat'
    VALIDATION1_CSV = 'SaukAtSauk_WY2009_validation.csv'
    VALIDATION2_CSV = 'AbvWhitechuck_WY2009_validation.csv'
    DHSVM_CMD = '/mnt/d/keckje/dhsvm/projects/sauk/Fast_CA_WilcoxVersion/sourcecode_cp/DHSVM3.1.3 ' + CONFIG_FILE 
    DIR_PREFIX = "dhsvm_run_pnnl_data_pid_"

    #create blank csv files to record metrics of model run
    writeblankcsv('sim1.csv')
    writeblankcsv('sim2.csv')	
    
    # Initialize the Dream Class
    fast_run = fast_run_setup()
	
    sampler=spotpy.algorithms.fast(fast_run, dbname= DB_NAME, dbformat='sql', parallel='mpc',save_sim=False)
    sampler.sample(NUM_OF_ITERATIONS)
    results = sampler.getdata()
    plot_results(results)
