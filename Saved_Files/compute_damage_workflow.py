# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:14:22 2024

@author: Azelice Ludot - April 2024 

PURPOSE: This code is automatizing the run of a given number of OpenFAST simulations and post-processing of the mooring line tensions
    
INPUT: Input conditions 

OUTPUT: Database of damage value corresponding to the nput metocean conditions
"""

###############################################################################
#%% Import packages

# System 
import time
import os
# Data
import numpy as np
import math
import csv
import pandas as pd
from openfast_toolbox.io import FASTOutputFile
from openfast_toolbox.postpro import equivalent_load


nb_ML = 4 
nb_segments = 0

###############################################################################
#%% CREATE RESULTS FILES
###############################################################################

# Iterate through each mooring line
for mooring_line in range (1, nb_ML+1):
    # Create a CSV file for each segment and each corrosion grade
    for segment in range (0, nb_segments+2):
        for corrosion in range (1,8):
            csv_file_path = f'ML{mooring_line}_Segment{segment}_Corrosion{corrosion}_results.csv'
            print(csv_file_path)
            with open(os.getcwd()+'/Results/'+csv_file_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                # Write headers or any data you want
                writer.writerow(["U10 [m/s]", "Hs [m]", "Tp [s]", "theta_wind [deg]", "theta_mis [deg]", "DEL [-]"])  # Example headers
    
            print(f"CSV file '{csv_file_path}' created.")

      
        
###############################################################################
#%% COMPUTE DAMAGE EQUIVALENT LOAD  
###############################################################################

# Read an openFAST binary
fastoutFilename = os.path.join(scriptDir, '../../../data/example_files/fastout_allnodes.outb')
df = FASTOutputFile(fastoutFilename).toDataFrame()


# Compute equivalent load for one signal and Wohler slope
m = 3 # Wohler slope
Leq = equivalent_load(df['Time_[s]'], df['RootMyc1_[kN-m]'], m=m)

    
    for i in range(4): # Compute damage for the 4 fairleads

        CPU_name = 'CPU'+str(i+1)
        
        path_results = CPU_name+'/IEA-15-240-RWT-UMaineSemi'
        
        transient_time = 1000
                
        [Damage_ML1, Damage_ML2, Damage_ML3, Damage_ML4]=damage_computation(path_results, segments, transient_time, MBL, corrosion, cross_section_area, m_SN)
        
        for segment in range(0, nb_segments+2):
            for line in range(1, nb_ML+1):  # 3 mooring lines
                filename = f'ML{line}_Segment{segment}_Corrosion{corrosion}_results.csv'
                with open(os.getcwd()+'/Results/'+filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    
                    if line == 1:
                        writer.writerow([parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3],parameters[i,4], Damage_ML1[segment]])  # Write subset and damage to CSV
                    elif line == 2:
                        writer.writerow([parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3],parameters[i,4], Damage_ML2[segment]])  # Write subset and damage to CSV
                    elif line == 3:
                        writer.writerow([parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3],parameters[i,4], Damage_ML3[segment]])  # Write subset and damage to CSV

    ## Delete the files 
    for i in range(nb_CPU):
        CPU_name = 'CPU'+str(i+1)
        folder_path = os.getcwd()+'/' + CPU_name  
        delete_folder(folder_path + '/IEA-15-240-RWT')
        delete_folder(folder_path + '/IEA-15-240-RWT-UMaineSemi')
    
    # Define the directory to search for files. Use '.' for the current directory.
    directory = '.'
    
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the filename ends with .out or .err
        if filename.endswith(".out"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Delete the file
            os.remove(file_path)
            # Print a confirmation message
            print(f"Deleted {file_path}")
    

    print("--- One simulation running time : %s seconds ---" % (time.time() - start1_time)) #Running time for a single simulation 

"""
END OF THE SCRIPT
"""       
