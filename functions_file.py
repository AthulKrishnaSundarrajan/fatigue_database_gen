# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:45:03 2024

@author: AzéliceLudot
"""

import numpy as np
import csv
import os  # to get the files of the directory
from openfast_toolbox.io import FASTOutputFile
#from pyFAST.input_output import FASTOutputFile
from openfast_toolbox.tools.fatigue import eq_load_and_cycles
from hipersim.turbgen.generate_field import generate_field

import shutil


def read_input_csv(file_path):
    inputs = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                key, value = row
                inputs[key.strip()] = value.strip()
    return inputs

def delete_folder(folder_path):
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        # Delete the folder
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def copy_folder(source_dir, destination_dir):
    try:
        # Check if source directory exists
        if not os.path.exists(source_dir):
            print(f"Source directory '{source_dir}' does not exist.")
            return

        # Create destination directory if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Copy the contents of source directory to destination directory
        shutil.copytree(source_dir, os.path.join(destination_dir, os.path.basename(source_dir)))
        print(f"Folder '{source_dir}' copied to '{destination_dir}' successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        

def Mann_model(Umean, Nx, Ny, Nz, dx, dy, dz, Tsim, BaseName, SeedNo):
    """
    Link to the documentation of the model : https://gitlab.windenergy.dtu.dk/HiperSim/hipersim/-/blob/master/examples/Turbgen/Turbgen_Manual_0.0.24.ipynb
    
    """

    # Set up parameters for the Mann model
    L = 29.4
    Gamma = 3.9
    alphaepsilon = 1

    HighFreqComp = 1

    [u, v, w] = generate_field(
        BaseName,
        alphaepsilon,
        L,
        Gamma,
        SeedNo,
        Nx,
        Ny,
        Nz,
        dx,
        dy,
        dz,
        HighFreqComp,
        SaveToFile=1,
    )
    
    ## Option: generate multiple turbulence boxes with identical paremeters but different seeds. This 
    ## approach saves some computation time because the Fourier coefficients are generated only once 
    ## (hence, the more seeds are used simultaneously, the bigger benefit)
    
    # SeedNo=[10, 12, 145]
    
    # T=turbgen.turb_field(
    #     BaseName,
    #     alphaepsilon,
    #     L,
    #     Gamma,
    #     SeedNo,
    #     Nx,
    #     Ny,
    #     Nz,
    #     dx,
    #     dy,
    #     dz,
    #     HighFreqComp,
    #     SaveToFile=1,
    # )
    
    # [u,v,w]=T.generate()
    

def move_files(file_names, source_folder, destination_folder):
    source_folder = os.getcwd()  # Get the current working directory
    # Check if the destination folder exists
    if not os.path.exists(destination_folder):
        print(f"Destination folder '{destination_folder}' does not exist.")
        return

    # Iterate over provided file names
    for file_name in file_names:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        # Check if the file exists in the source folder
        if os.path.exists(source_file_path):
            # Move the file to the destination folder
            shutil.move(source_file_path, destination_file_path)
            print(f"Moved '{file_name}' to '{destination_folder}'.")
        else:
            print(f"File '{file_name}' does not exist in the source folder.")
            
            
def damage(values, m_SN, a_SN, cross_section_area):

    eq_loads, cycles, ampl_bin_mean_tension, ampl_bin_edges = eq_load_and_cycles(
        values, no_bins=100, m=m_SN, neq=10 ** 7
    )

    cycles = [x for x in cycles if x != 0]  # removing values with 0 cycles
    ampl_bin_mean_stress= [
        x / cross_section_area for x in ampl_bin_mean_tension if str(x) != "nan"
    ]  # removing amplitude with no cycles and converting in stress (MPa)
    
    ampl_bin_mean_tension_kN= [
        x / 10**3 for x in ampl_bin_mean_tension if str(x) != "nan"
    ]  # removing amplitude with no cycles and converting in kN
    
    D = 0

    for j in range(len(cycles)):  # computation of the damage
        D += (cycles[j] / a_SN) * ampl_bin_mean_stress[j] ** m_SN
    
    # plt.hist(ampl_bin_mean, cycles)
    # plt.xlabel('Tension [kN]')
    # plt.ylabel('Number of cycles')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(True, which='both')
    # plt.grid(which='major', linestyle='--', linewidth='0.5', color='gray')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    # plt.minorticks_on()
    # plt.show() 
    
    
    return D

def damage_computation(path_results, segments, transient_time, MBL, corrosion, cross_section_area, m_SN):

    from functions_file import damage

    # # Get current directory so this script can be called from any location
    # scriptDir = os.path.dirname(__file__)

    ###########################################################################
    # Read an openFAST output: Mooring Line 1 
    fastoutFilename_ML1 = os.path.join(
        path_results + "/IEA-15-240-RWT-UMaineSemi.MD.Line1.out"
    )
    df_ML1 = FASTOutputFile(fastoutFilename_ML1).toDataFrame()
    
    # Read an openFAST output: Mooring Line 2
    fastoutFilename_ML2 = os.path.join(
        path_results + "/IEA-15-240-RWT-UMaineSemi.MD.Line2.out"
    )
    df_ML2 = FASTOutputFile(fastoutFilename_ML2).toDataFrame()
    
    # Read an openFAST output: Mooring Line 3
    fastoutFilename_ML3 = os.path.join(
        path_results + "/IEA-15-240-RWT-UMaineSemi.MD.Line3.out"
    )
    df_ML3 = FASTOutputFile(fastoutFilename_ML3).toDataFrame()

    ###########################################################################
    fastoutFilename = os.path.join(path_results + "/IEA-15-240-RWT-UMaineSemi.MD.out")
    df = FASTOutputFile(fastoutFilename).toDataFrame()
    
    ###########################################################################

    transient_time_index = np.where(df["Time_[s]"] == transient_time)[0][0]

    ###########################################################################
    Damage_ML1 = np.zeros(
        2 + len(segments)
    )  # creation of the variable to store the Damage to the anchor/Fairlead + interesting segments
    Damage_ML2 = np.zeros(
        2 + len(segments)
    )  # creation of the variable to store the Damage to the anchor/Fairlead + interesting segments
    Damage_ML3 = np.zeros(
        2 + len(segments)
    )  # creation of the variable to store the Damage to the anchor/Fairlead + interesting segments

    ###########################################################################
    # MOORING LINE 1
    ###########################################################################
    mean_load_fairlead=np.mean(df["FAIRTEN1_[N]"][transient_time_index:].values)/MBL*100
    a_SN_fairlead=10**(11.904-0.0507*mean_load_fairlead-0.106*corrosion)
    
    Damage_ML1[0] = damage(
        df["FAIRTEN1_[N]"][transient_time_index:].values, m_SN, a_SN_fairlead, cross_section_area
    )  # Fairlead 1
   
    mean_load_anchor=np.mean(df["ANCHTEN1_[N]"][transient_time_index:].values)/MBL*100
    a_SN_anchor=10**(11.904-0.0507*mean_load_anchor-0.106*corrosion)
    
    Damage_ML1[1] = damage(
        df["ANCHTEN1_[N]"][transient_time_index:].values, m_SN, a_SN_anchor, cross_section_area
    )  # Anchor 1

    # Computation of damage and rainflow counting for segments within the ML1
    compt = 0
    for i in segments:
        
        mean_load_segment=np.mean(df_ML1[i + "_[N]"][transient_time_index:].values)/MBL*100
        a_SN_segment=10**(11.904-0.0507*mean_load_segment-0.106*corrosion)
        
        Damage_ML1[compt + 2] = damage(
            df_ML1[i + "_[N]"][transient_time_index:].values, m_SN, a_SN_segment, cross_section_area
        )
        compt += 1
    
    ###########################################################################
    # MOORING LINE 2
    ###########################################################################
    mean_load_fairlead=np.mean(df["FAIRTEN2_[N]"][transient_time_index:].values)/MBL*100
    a_SN_fairlead=10**(11.904-0.0507*mean_load_fairlead-0.106*corrosion)
    
    Damage_ML2[0] = damage(
        df["FAIRTEN2_[N]"][transient_time_index:].values, m_SN, a_SN_fairlead, cross_section_area
    )
   
    mean_load_anchor=np.mean(df["ANCHTEN2_[N]"][transient_time_index:].values)/MBL*100
    a_SN_anchor=10**(11.904-0.0507*mean_load_anchor-0.106*corrosion)
    
    Damage_ML2[1] = damage(
        df["ANCHTEN2_[N]"][transient_time_index:].values, m_SN, a_SN_anchor, cross_section_area
    )

    # Computation of damage and rainflow counting for segments within the ML2
    compt = 0
    for i in segments:
        
        mean_load_segment=np.mean(df_ML2[i + "_[N]"][transient_time_index:].values)/MBL*100
        a_SN_segment=10**(11.904-0.0507*mean_load_segment-0.106*corrosion)
        
        Damage_ML2[compt + 2] = damage(
            df_ML2[i + "_[N]"][transient_time_index:].values, m_SN, a_SN_segment, cross_section_area
        )
        compt += 1
        
    ###########################################################################
    # MOORING LINE 3
    ###########################################################################
    mean_load_fairlead=np.mean(df["FAIRTEN3_[N]"][transient_time_index:].values)/MBL*100
    a_SN_fairlead=10**(11.904-0.0507*mean_load_fairlead-0.106*corrosion)
    
    Damage_ML3[0] = damage(
        df["FAIRTEN3_[N]"][transient_time_index:].values, m_SN, a_SN_fairlead, cross_section_area
    )
   
    mean_load_anchor=np.mean(df["ANCHTEN3_[N]"][transient_time_index:].values)/MBL*100
    a_SN_anchor=10**(11.904-0.0507*mean_load_anchor-0.106*corrosion)
    
    Damage_ML3[1] = damage(
        df["ANCHTEN3_[N]"][transient_time_index:].values, m_SN, a_SN_anchor, cross_section_area
    )

    # Computation of damage and rainflow counting for segments within the ML2
    compt = 0
    for i in segments:
        
        mean_load_segment=np.mean(df_ML3[i + "_[N]"][transient_time_index:].values)/MBL*100
        a_SN_segment=10**(11.904-0.0507*mean_load_segment-0.106*corrosion)
        
        Damage_ML3[compt + 2] = damage(
            df_ML3[i + "_[N]"][transient_time_index:].values, m_SN, a_SN_segment, cross_section_area
        )
        compt += 1
        
        
    return (Damage_ML1, Damage_ML2, Damage_ML3)

def normalize_angle(angle):
    if angle < -180:
        angle += 360
    elif  angle >= 180:
        angle -= 360
    return angle
