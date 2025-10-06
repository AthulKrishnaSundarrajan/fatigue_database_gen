# -*- coding: utf-8 -*-
def _load_yaml(file_path):
    
    import yaml
    import os 
    
    """Loads a YAML file and returns its data."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist.")
        return {}

    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error reading {file_path}: {exc}")
            return {}

def create_results_file(nb_ML, list_segments_of_interest)

    import csv
    import os
    
    # Iterate through each mooring line
    for mooring_line in range (1, nb_ML+1):
        # Create a CSV file for each segment and each corrosion grade
        for segment in list_segments_of_interest:
            csv_file_path = segment + '_results.csv'
            print(csv_file_path)
            with open(os.getcwd()+'/Results/'+csv_file_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                # Write headers or any data you want
                writer.writerow(["Design number", "U_hub [m/s]", "Hs [m]", "Tp [s]", "theta_wind [deg]", "theta_mis [deg]", "d_chain", "d_rope", "L_rope", "radius", "z_fairlead", "seed_wind", "seed_wave", "DEL [-]"])  # Example headers

            print(f"CSV file '{csv_file_path}' created.")
            
def compute_DEL(scriptDir, name_simulation, file_out, input_file): 
    
    #%%###############################################################################
    # Import packages

    # System 
    import time
    import os
    # Data
    import numpy as np
    import math
    import csv
    import pandas as pd
    from openfast_toolbox.io import FASTOutputFile
    from openfast_toolbox.tools.fatigue import find_range_count
    import rainflow 

    input_data = _load_yaml(input_file)
    number_simulation = int(name_simulation.split("_")[1])
    
    # Search for the experiment with that ID
    experiment = next(
        (exp for exp in input_data["experiments"] if exp["experiment_id"] == number_simulation),
        None
    )

    if experiment:
        print("Found experiment:", experiment)
        l_rope = experiment["L_rope"]
        d_chain = experiment["d_chain"]
        d_rope = experiment["d_rope"]
        radius = experiment["radius"]
        z_fairlead = experiment["z_fairlead"]
        wind_speed = experiment["hub_wind_speed"]
        significant_wave_height = experiment["significant_wave_height"]
        peak_period = experiment["peak_period"]
    else:
        print("Experiment not found.")
    
    nb_ML = 4 
    list_segments_of_interest = ["FairTen1", "FairTen2", "FairTen3",
                                "FairTen4", "FairTen5", "FairTen6",
                                "FairTen7", "FairTen8", "FairTen9",
                                "FairTen10", "FairTen11", "FairTen12"]

        
            
    #%%###############################################################################
    # COMPUTE DAMAGE EQUIVALENT LOAD  
    ###############################################################################

    # Read an openFAST binary
    fastoutFilename = os.path.join(scriptDir, file_out)
    df = FASTOutputFile(fastoutFilename).toDataFrame()
    
    transient_time = 1000
    transient_time_index_out = np.where(df["Time_[s]"] == transient_time)[0][0]


    # Compute equivalent load for one signal and Wohler slope
    for segment in list_segments_of_interest:
        if segment in ["FairTen2", "FairTen5", "FairTen8"]:
            m = 3 # Wohler slope
            a = 6.0 * 10**6 # intercept parameter
            area = (np.pi / 4) * d_chain ** 2
            # --- TENSION TO STRESS ---
            stress = df[segment+'_[N]'] / area
            # --- RAINFLOW COUNTING ---
            cycles = rainflow.count_cycles(stress)
            # --- DAMAGE CALCULATION ---
            damage = 0.0
            for range_val, count in cycles:
                if range_val <= 0:
                    continue  # Skip invalid cycles
                N = a * (range_val) ** (-m)  # Wöhler curve: cycles to failure
                damage += count / N          # Miner’s rule: accumulated damage
        else:
            m = 13.46 # Wohler slope
            a = 0.259 # interceptparameter
            # --- RAINFLOW COUNTING ---
            cycles = rainflow.count_cycles(df[segment+'_[N]'])
            # --- DAMAGE CALCULATION ---
            damage = 0.0
            for range_val, count in cycles:
                if range_val <= 0:
                    continue  # Skip invalid cycles
                N = a * (range_val) ** (-m)  # Wöhler curve: cycles to failure
                damage += count / N          # Miner’s rule: accumulated damage
                
        filename = segment + '_results.csv'
        with open(os.getcwd()+'/Results/'+filename, mode='a', newline='') as file:
            writer = csv.writer(file)
    
            writer.writerow([number_simulation, wind_speed, significant_wave_height, peak_period, 0, 0, d_chain, d_rope, l_rope, radius, z_fairlead, 0, 0, damage])  # Write subset and damage to CSV



def compute_damage(timeseries, m, a): 
    
    #%%###############################################################################
    # Import packages

    # System 
    import time
    import os
    # Data
    import numpy as np
    import math
    import csv
    import pandas as pd
    import rainflow


    # --- RAINFLOW COUNTING ---
    cycles = rainflow.count_cycles(timeseries)
    # --- DAMAGE CALCULATION ---
    damage = 0.0
    for range_val, count in cycles:
        if range_val <= 0:
            continue  # Skip invalid cycles
        N = a * (range_val) ** (-m)  # Wöhler curve: cycles to failure
        damage += count / N          # Miner’s rule: accumulated damage

    return damage

if __name__ == "__main__":
    scriptDir = 
    name_simulation = 
    compute_DEL(scriptDir, name_simulation, file_out = 'sample1.outb', input_file = "../Inputs_Definition/DOE_Design_Environmental_Definitions.yaml")