# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:49:02 2023

@author: Gwang-Noh Ahn
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import logging
import pycont 
import threading
import pyautogui
import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
from tbwk.RawOpener import unpack, Block
from tbwk.Measurement import Measurement


#define pumps 
logging.basicConfig(level=logging.INFO)

# simply import the module


# link to your config file
SETUP_CONFIG_FILE = 'config.json'

# and load the config file in a MultiPumpController
controller = pycont.controller.MultiPumpController.from_configfile(SETUP_CONFIG_FILE)

# initialize the pumps in a smart way, if they are already initialized we do not want to reinitialize them because they go back to zero position
controller.smart_initialize()

#change required flowrate of the pump to the speed of the pump (12500 is considered for the volume of the syringe 12.5 mL and it can be changed)
def speed(flowrate):   #ml/min
    ulpers = flowrate*(100/6)
    return round((48000/12500)*ulpers)

#initialize the pump (one pump)
def initializer(pump):
    controller.pumps[pump].initialize(valve_position="E")
    
    
#initialize all the pumps   
def init_all(washer = False ):
    aprocess1 = threading.Thread(target=initializer,args=("dox",))
    aprocess2 = threading.Thread(target=initializer,args=("lip",))
    aprocess4 = threading.Thread(target=initializer,args=("inter",))
    aprocess1.start()
    aprocess2.start()
    aprocess4.start()


def init_all1(washer = False ):
    aprocess1 = threading.Thread(target=initializer,args=("dox",))
    aprocess2 = threading.Thread(target=initializer,args=("lip",))
    aprocess1.start()
    aprocess2.start()

def fill(pump,valve):
    spd = 5000
    controller.pumps[pump].set_valve_position(valve)
    controller.pumps[pump].go_to_volume(12.5, wait=True,speed=spd)
    controller.pumps[pump].go_to_volume(9.5, wait=True,speed=spd)
    controller.pumps[pump].go_to_volume(12.5, wait=True,speed=spd)
    
def fill_all():
    thread1 = threading.Thread(target = fill, args = ("dox","E"))
    thread2 = threading.Thread(target = fill, args = ("lip","E"))
    thread1.start()
    thread2.start()

    
def unload(pump,valve):
    spd = 5500
    controller.pumps[pump].set_valve_position(valve)
    controller.pumps[pump].go_to_volume(0, wait=True,speed=spd)

def unload_all(valve):
    thread1 = threading.Thread(target = unload, args = ("dox",valve))
    thread2 = threading.Thread(target = unload, args = ("lip",valve))
    thread1.start()
    thread2.start()


#synthesize the nanoparticles
def synt(tfr,r):
    v = 3
    lip_fr = tfr/(r+1)
    dox_fr = (r*tfr)/(r+1)
    fill_all()
    time.sleep(30)
    controller.pumps["dox"].set_valve_position("I")
    controller.pumps["lip"].set_valve_position("I")
    controller.pumps["inter"].set_valve_position("O")
    controller.pumps["dox"].go_to_volume(0, wait=False,speed=speed(dox_fr))
    controller.pumps["lip"].go_to_volume(0, wait=False,speed=speed(lip_fr))
    controller.pumps["inter"].go_to_volume(12.5, wait=False,speed=speed(tfr))
    time.sleep((v/tfr)*60)
    controller.pumps["dox"].terminate()
    controller.pumps["lip"].terminate()
    controller.pumps["inter"].terminate()
    init_all1()

#encapsulation efficiency from the uv vis
def ee(absorbance,r):
    up = absorbance*8*1.8/22.28
    down = (r/(r+1))*0.1
    val = up/down
    
    return val


def uv_measure(repeat):
    
    full_abs =[]
    for i in range(repeat):
        
        path = 'kinetics.twbk'
        
        
        with open(path, "rb") as fh:
            content = fh.read()
            
        
        block = unpack(content)
        
        
        types = []
        
        for i in block:
            types.append(i.type)
        
        abss = []   
         
        # block_ = block[90]
        # wave_length = block_.parsed_content[1].parsed_content[2].parsed_content[2].parsed_content[3]
        
        for i in range(len(block)):
            blockser = block[i]
            if blockser.type ==151: 
                absorption = blockser.parsed_content[1].parsed_content[2].parsed_content[1].parsed_content[3]
                abss.append(absorption)
        full_abs.append(abss[-1][295])
        time.sleep(10)
            
    avg = np.average(full_abs)

    return avg

def wait(time1): #time in second 
    amount = int(time1/10)
    for i in range(amount):
        time.sleep(10)



def main(tfr,r,run):
    synt(tfr, r)
    controller.pumps["inter"].set_valve_position("I")
    controller.pumps["inter"].go_to_volume(2, wait=True,speed=speed(1))
    controller.pumps["inter"].go_to_volume(0, wait=False,speed=speed(0.05))
    wait(20*60)  #wait for 20 mins 
    absorbance = uv_measure(10)
    encaps = ee(absorbance,r)
    controller.pumps["inter"].terminate()
    size,pdi = measure(run)
    controller.pumps["inter"].go_to_volume(0, wait=True,speed=speed(5))
    wash()
    
    return encaps,size,pdi
    
    
def wash():
    controller.pumps["inter"].set_valve_position("E")
    controller.pumps["inter"].go_to_volume(3, wait=True,speed=5000)
    controller.pumps["inter"].set_valve_position("I")
    controller.pumps["inter"].go_to_volume(0, wait=True,speed=speed(2))
    controller.pumps["inter"].set_valve_position("E")
    controller.pumps["inter"].go_to_volume(3, wait=True,speed=5000)
    controller.pumps["inter"].set_valve_position("I")
    controller.pumps["inter"].go_to_volume(0, wait=False,speed=speed(2))
    
    
    

    

    
    
    
def inject(pump,valve,volume,flowrate,wait):
    spd = speed(flowrate)
    controller.pumps[pump].deliver(volume,valve,spd,wait = wait)
    
def warm_up():
    #put the inlet of polymer to the solvent 
    fill_all()
    time.sleep(9.5)
    unload_all("O")
    time.sleep(10)
    fill_all()
    time.sleep(9.5)
    unload_all("O")
    time.sleep(10)
        
def measure(run): #measuring the size and pdi in the dls device'
    pyautogui.moveTo(1142,56)
    time.sleep(1)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.click()
    #run a macro code for runing the analysis
    while True:
        try:
            df = pd.read_csv("first.txt",header=None).to_numpy()
            size = np.average(df[run,0])
            pdi = np.average(df[run,1])
            break
        except Exception as e:
            time.sleep(5)
        
    return size,pdi

def wash():
    
    


def flow(x,run):
    tfr = x[0]
    r = x[1]
    c = x[2]
    target = 240
    v = 2
    pol_fr = tfr/(r+1)
    water_fr = (r*tfr)/(r+1)
    purepol_fr = (c/10)*pol_fr
    dil_fr = (1-c/10)*pol_fr
    fill_all()
    time.sleep(9.5)
    inject("pol","I",v+0.1,purepol_fr,False)
    inject("dil","I",v+0.1,dil_fr,False)
    pause = (0.1/pol_fr)*(1/60)  #time for waiting to polymer to reach to antisolvent
    time.sleep(pause)
    inject("water","I",v,water_fr,True)
    controller.pumps["pol"].terminate()
    controller.pumps["dil"].terminate()
    init_all(washer=False)
    size,pdi= measure(run)
    print(f'Size = {size :.2f}\nPDI = {pdi:.2f}')
    wash(3)
    return np.abs(size-target)
  
def flow_nm(tfr,r,c):
    target = 137
    v = 1
    pol_fr = tfr/(r+1)
    water_fr = (r*tfr)/(r+1)
    purepol_fr = (c/10)*pol_fr
    dil_fr = (1-c/10)*pol_fr
    fill_all()
    time.sleep(9.5)
    inject("pol","I",v+0.1,purepol_fr,False)
    inject("dil","I",v+0.1,dil_fr,False)
    pause = (0.6/pol_fr)*(1/60)  #time for waiting to polymer to reach to antisolvent
    time.sleep(pause)
    inject("water","I",v,water_fr,True)
    controller.pumps["pol"].terminate()
    controller.pumps["dil"].terminate()
    init_all(washer=False)
 
from skopt.benchmarks import branin as _branin
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from matplotlib.colors import LogNorm
from skopt import gp_minimize
from skopt.optimizer import Optimizer
from sklearn.metrics import r2_score
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.neighbors import KNeighborsRegressor
import random
import math
import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt.optimizer import Optimizer 
import pandas as pd 


matern_fixed = ConstantKernel(1.0, constant_value_bounds='fixed') * Matern(
    length_scale=np.ones(1), length_scale_bounds='fixed', nu=2.5)
matern_tunable = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)

bounds = [(1,20),(1,15),(2.5,2.6)]    
#defining regressors 
gpr = GaussianProcessRegressor(kernel=matern_fixed)   

tfrs= []
rs = []
cs = []
difs = []
opter_min =  Optimizer(bounds,base_estimator=gpr,n_initial_points=3,acq_func="EI",random_state=np.random.randint(1200))
gap = 10
i=0
dif = 1000
while dif>gap:
    print(i)
    asked0 = opter_min.ask()
    dif = flow(asked0,run=0)
    opter_min.tell(asked0,dif)
    print(asked0)
    print(dif)
    tfrs.append(asked0[0])
    rs.append(asked0[1])
    cs.append(asked0[2])
    difs.append(dif)
    dc = {"tfr":tfrs,"r":rs,"c":cs,"dif":difs}
    df = pd.DataFrame(dc)
    df.to_csv("round"+str(i)+".csv")
    i=i+1
    print("round finished")
    
    



reaction_components = {
    'c': np.arange(1,10.1,0.2),
    'r': np.arange(1,15,1),
    'tfr': np.arange(1,25.5,1),
}




controller.pumps["inter"].current_volume
