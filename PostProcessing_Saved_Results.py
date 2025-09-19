import os
import sys
import numpy as np
import csv
import pandas as pd
from openfast_toolbox.io import FASTOutputFile, FASTInputFile
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_yaml(file_path):
    
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

def list_folders(directory):
    """Returns a list of folders in the given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

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

def compute_DEL(signal, m, fs=1.0):
    """
    Compute the Damage Equivalent Load (DEL) from a signal using rainflow counting.

    Parameters:
        signal (array-like): The input load/time series.
        m (float): Wöhler exponent.
        fs (float): Sampling frequency (Hz). Default is 1.0.

    Returns:
        float: The computed DEL value.
    """
    import rainflow

    cycles = rainflow.extract_cycles(signal)
    # cycles: list of tuples (range, mean, count, start_index, end_index)
    damage = sum(count * (rng ** m) for rng, mean, count, start, end in cycles)
    N_eq = len(signal) / fs  # Equivalent number of cycles

    return (damage / N_eq) ** (1 / m)



#def analyze_simulation_results(sim_name, seeds_folders, current_folder):
def analyze_simulation_results(file_settings, sim_folder, current_folder, seeds_folders, input_file, main_file_name):
    """
    Analyze OpenFAST simulation results for floater, WTG, and mooring requirements.

    Parameters:
        seeds_folder (list): List of names of the seeds folders where the sample was ran. 
        sim_name (str): Name of the simulation to be analyzed.
    """
    
    fixed_variables = load_yaml(file_settings)

    depth = fixed_variables["mooring_layout"]["depth"]
    
    results_folder = "../Results_test_2"
    os.makedirs(results_folder, exist_ok=True)
    csv_filename = f"{sim_folder}_Results.csv"
    csv_path = os.path.join(results_folder, csv_filename)
    
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Wind Speed (m/s)",
                "Significant Wave Height (m)",
                "Wave Period (s)",
                "Wind Direction (deg)",
                "Wave Direction (deg)",
                "Length rope (-)", 
                "Chain diameter (mm)",
                "Rope diameter (mm)", 
                "Mooring radius (m)",
                "Fairlead verticalposition (m)",
                "Mean Pitch Angle (deg)",
                "Max Pitch Angle (deg)",
                "Mean Roll Angle (deg)",
                "Max Roll Angle (deg)",
                "Max Vertical Acceleration (m/s²)",
                "Max Horizontal Acceleration (m/s²)",
                "Max Yaw Angle (deg)",
                "Max Surge Excursion (m)",
                "Max Sway Excursion (m)",
                "Max Tension at Top Chain Fairlead Line 1 (N)",
                "Max Tension at Top Chain Fairlead Line 2 (N)",
                "Max Tension at Top Chain Fairlead Line 3 (N)",
                "Max Tension at Top Chain Fairlead Line 4 (N)",
                "Max Tension at Rope Fairlead Line 1 (N)",
                "Max Tension at Rope Fairlead Line 2 (N)",
                "Max Tension at Rope Fairlead Line 3 (N)",
                "Max Tension at Rope Fairlead Line 4 (N)",
                "Max Tension at Bottom Chain Fairlead Line 1 (N)",
                "Max Tension at Bottom Chain Fairlead Line 2 (N)",
                "Max Tension at Bottom Chain Fairlead Line 3 (N)",
                "Max Tension at Bottom Chain Fairlead Line 4 (N)",
                "Mean Tension at Top Chain Fairlead Line 1 (N)",
                "Mean Tension at Top Chain Fairlead Line 2 (N)",
                "Mean Tension at Top Chain Fairlead Line 3 (N)",
                "Mean Tension at Top Chain Fairlead Line 4 (N)",
                "Mean Tension at Rope Fairlead Line 1 (N)",
                "Mean Tension at Rope Fairlead Line 2 (N)",
                "Mean Tension at Rope Fairlead Line 3 (N)",
                "Mean Tension at Rope Fairlead Line 4 (N)",
                "Mean Tension at Bottom Chain Fairlead Line 1 (N)",
                "Mean Tension at Bottom Chain Fairlead Line 2 (N)",
                "Mean Tension at Bottom Chain Fairlead Line 3 (N)",
                "Mean Tension at Bottom Chain Fairlead Line 4 (N)",
                "90th percentile Tension at Top Chain Fairlead Line 1 (N)",
                "90th percentile Tension at Top Chain Fairlead Line 2 (N)",
                "90th percentile Tension at Top Chain Fairlead Line 3 (N)",
                "90th percentile Tension at Top Chain Fairlead Line 4 (N)",
                "90th percentile Tension at Rope Fairlead Line 1 (N)",
                "90th percentile Tension at Rope Fairlead Line 2 (N)",
                "90th percentile Tension at Rope Fairlead Line 3 (N)",
                "90th percentile Tension at Rope Fairlead Line 4 (N)",
                "90th percentile Tension at Bottom Chain Fairlead Line 1 (N)",
                "90th percentile Tension at Bottom Chain Fairlead Line 2 (N)",
                "90th percentile Tension at Bottom Chain Fairlead Line 3 (N)",
                "90th percentile Tension at Bottom Chain Fairlead Line 4 (N)",
                "Damage at Top Chain Fairlead Line 1 (N)",
                "Damage at Top Chain Fairlead Line 2 (N)",
                "Damage at Top Chain Fairlead Line 3 (N)",
                "Damage at Top Chain Fairlead Line 4 (N)",
                "Damage at Rope Fairlead Line 1 (N)",
                "Damage at Rope Fairlead Line 2 (N)",
                "Damage at Rope Fairlead Line 3 (N)",
                "Damage at Rope Fairlead Line 4 (N)",
                "Damage at Bottom Chain Fairlead Line 1 (N)",
                "Damage at Bottom Chain Fairlead Line 2 (N)",
                "Damage at Bottom Chain Fairlead Line 3 (N)",
                "Damage at Bottom Chain Fairlead Line 4 (N)", 
                "DEL at Top Chain Fairlead Line 1 (N)",
                "DEL at Top Chain Fairlead Line 2 (N)",
                "DEL at Top Chain Fairlead Line 3 (N)",
                "DEL at Top Chain Fairlead Line 4 (N)",
                "DEL at Rope Fairlead Line 1 (N)",
                "DEL at Rope Fairlead Line 2 (N)",
                "DEL at Rope Fairlead Line 3 (N)",
                "DEL at Rope Fairlead Line 4 (N)",
                "DEL at Bottom Chain Fairlead Line 1 (N)",
                "DEL at Bottom Chain Fairlead Line 2 (N)",
                "DEL at Bottom Chain Fairlead Line 3 (N)",
                "DEL at Bottom Chain Fairlead Line 4 (N)"
            ])  # Example headers
    
    for seed in seeds_folders:
    
        # Define file paths for each seed folder
        fastout_filename_out = os.path.join(current_folder + '/' + 'Output_' + sim_folder + '_' + seed + '.out')
        
        # Read data
        df_out = FASTOutputFile(fastout_filename_out).toDataFrame()
        
        # === Design parameters
        input_data = load_yaml(input_file)
        number_simulation = int(sim_folder.split("_")[1])
        
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
        
        #== Transient time
        transient_time = 1000
        transient_time_index_out = np.where(df_out["Time_[s]"] == transient_time)[0][0]

        # === Floater Requirements ===
        mean_pitch_angles = np.mean(df_out['PtfmPitch_[deg]'][transient_time_index_out:])
        max_pitch_angles = np.max(np.abs(df_out['PtfmPitch_[deg]'][transient_time_index_out:]))
        mean_roll_angles = np.mean(df_out['PtfmRoll_[deg]'][transient_time_index_out:])
        max_roll_angles = np.max(np.abs(df_out['PtfmRoll_[deg]'][transient_time_index_out:]))

        # === Wind Turbine Generator (WTG) Requirements ===
        max_vertical_accs = np.max(np.abs(df_out['NcIMUTAzs_[m/s^2]'][transient_time_index_out:]))
        max_horizontal_accs = np.max(
            np.maximum(
                np.abs(df_out['NcIMUTAxs_[m/s^2]'][transient_time_index_out:]),
                np.abs(df_out['NcIMUTAys_[m/s^2]'][transient_time_index_out:])
            )
        )
        max_yaw_ptfms = np.max(np.abs(df_out['PtfmYaw_[deg]'][transient_time_index_out:]))

        # === Mooring Requirements ===
        floater_exursions_surge = np.max(np.abs(df_out['PtfmSurge_[m]'][transient_time_index_out:]))
        floater_exursions_sway = np.max(np.abs(df_out['PtfmSway_[m]'][transient_time_index_out:]))
    
        # Max tensions
        max_tensions_line_1_fairlead_top_chain = np.max(np.abs(df_out["FAIRTEN1_[N]"][transient_time_index_out:]))
        max_tensions_line_2_fairlead_top_chain = np.max(np.abs(df_out["FAIRTEN4_[N]"][transient_time_index_out:]))
        max_tensions_line_3_fairlead_top_chain = np.max( np.abs(df_out["FAIRTEN7_[N]"][transient_time_index_out:]))
        max_tensions_line_4_fairlead_top_chain = np.max(np.abs(df_out["FAIRTEN10_[N]"][transient_time_index_out:]))
        
        max_tensions_line_1_fairlead_rope = np.max(np.abs(df_out["FAIRTEN2_[N]"][transient_time_index_out:]))
        max_tensions_line_2_fairlead_rope = np.max(np.abs(df_out["FAIRTEN5_[N]"][transient_time_index_out:]))
        max_tensions_line_3_fairlead_rope = np.max( np.abs(df_out["FAIRTEN8_[N]"][transient_time_index_out:]))
        max_tensions_line_4_fairlead_rope = np.max(np.abs(df_out["FAIRTEN11_[N]"][transient_time_index_out:]))
        
        max_tensions_line_1_fairlead_bottom_chain = np.max(np.abs(df_out["FAIRTEN3_[N]"][transient_time_index_out:]))
        max_tensions_line_2_fairlead_bottom_chain = np.max(np.abs(df_out["FAIRTEN6_[N]"][transient_time_index_out:]))
        max_tensions_line_3_fairlead_bottom_chain = np.max( np.abs(df_out["FAIRTEN9_[N]"][transient_time_index_out:]))
        max_tensions_line_4_fairlead_bottom_chain = np.max(np.abs(df_out["FAIRTEN12_[N]"][transient_time_index_out:]))
        
        # Mean tensions 
        mean_tensions_line_1_fairlead_top_chain = np.mean(df_out["FAIRTEN1_[N]"][transient_time_index_out:])
        mean_tensions_line_2_fairlead_top_chain = np.mean(df_out["FAIRTEN4_[N]"][transient_time_index_out:])
        mean_tensions_line_3_fairlead_top_chain = np.mean(df_out["FAIRTEN7_[N]"][transient_time_index_out:])
        mean_tensions_line_4_fairlead_top_chain = np.mean(df_out["FAIRTEN10_[N]"][transient_time_index_out:])
        
        mean_tensions_line_1_fairlead_rope = np.mean(df_out["FAIRTEN2_[N]"][transient_time_index_out:])
        mean_tensions_line_2_fairlead_rope = np.mean(df_out["FAIRTEN5_[N]"][transient_time_index_out:])
        mean_tensions_line_3_fairlead_rope = np.mean(df_out["FAIRTEN8_[N]"][transient_time_index_out:])
        mean_tensions_line_4_fairlead_rope = np.mean(df_out["FAIRTEN11_[N]"][transient_time_index_out:])
        
        mean_tensions_line_1_fairlead_bottom_chain = np.mean(df_out["FAIRTEN3_[N]"][transient_time_index_out:])
        mean_tensions_line_2_fairlead_bottom_chain = np.mean(df_out["FAIRTEN6_[N]"][transient_time_index_out:])
        mean_tensions_line_3_fairlead_bottom_chain = np.mean(df_out["FAIRTEN9_[N]"][transient_time_index_out:])
        mean_tensions_line_4_fairlead_bottom_chain = np.mean(df_out["FAIRTEN12_[N]"][transient_time_index_out:])
        
        # 90 percentile
        percentile_tensions_line_1_fairlead_top_chain = np.percentile(df_out["FAIRTEN1_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_2_fairlead_top_chain = np.percentile(df_out["FAIRTEN4_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_3_fairlead_top_chain = np.percentile(df_out["FAIRTEN7_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_4_fairlead_top_chain = np.percentile(df_out["FAIRTEN10_[N]"][transient_time_index_out:], 90)
        
        percentile_tensions_line_1_fairlead_rope = np.percentile(df_out["FAIRTEN2_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_2_fairlead_rope = np.percentile(df_out["FAIRTEN5_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_3_fairlead_rope = np.percentile(df_out["FAIRTEN8_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_4_fairlead_rope = np.percentile(df_out["FAIRTEN11_[N]"][transient_time_index_out:], 90)
        
        percentile_tensions_line_1_fairlead_bottom_chain = np.percentile(df_out["FAIRTEN3_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_2_fairlead_bottom_chain = np.percentile(df_out["FAIRTEN6_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_3_fairlead_bottom_chain = np.percentile( df_out["FAIRTEN9_[N]"][transient_time_index_out:], 90)
        percentile_tensions_line_4_fairlead_bottom_chain = np.percentile(df_out["FAIRTEN12_[N]"][transient_time_index_out:], 90)
        
        ## COMPUTE damage 
        
        # ---- WOHLER COEFFICIENTS ----
        m_chain = 3 # Wohler slope # DNV-OS-E301 (6.5.1)
        a_chain = 6.0 * 10**10 # intercept parameter # DNV-OS-E301 (6.5.1)
        m_rope = 13.46 # Wohler slope # DNV-OS-E301 (6.9.4)
        a_rope = 0.259 # interceptparameter # DNV-OS-E301 (6.9.4)
        area_chain = (2* np.pi / 4) * d_chain ** 2 # DNV-OS-E301 (6.2.10)
        MBS_rope = (0.293466*d_rope**2 + 0.206689*d_rope + 342.624548)*1000 # See Excel sheet for calculations # in N
        
        # --- TENSION TO STRESS ---
        stress_line_1_fairlead_top_chain = df_out["FAIRTEN1_[N]"][transient_time_index_out:] / area_chain
        stress_line_2_fairlead_top_chain = df_out["FAIRTEN4_[N]"][transient_time_index_out:] / area_chain
        stress_line_3_fairlead_top_chain = df_out["FAIRTEN7_[N]"][transient_time_index_out:] / area_chain
        stress_line_4_fairlead_top_chain = df_out["FAIRTEN10_[N]"][transient_time_index_out:] / area_chain
        
        stress_line_1_fairlead_bottom_chain = df_out["FAIRTEN3_[N]"][transient_time_index_out:] / area_chain
        stress_line_2_fairlead_bottom_chain = df_out["FAIRTEN6_[N]"][transient_time_index_out:] / area_chain
        stress_line_3_fairlead_bottom_chain = df_out["FAIRTEN9_[N]"][transient_time_index_out:] / area_chain
        stress_line_4_fairlead_bottom_chain = df_out["FAIRTEN12_[N]"][transient_time_index_out:] / area_chain
        
        # --- DAMAGE COMPUTATION -------
        
        damage_line_1_fairlead_top_chain = compute_damage(stress_line_1_fairlead_top_chain, m_chain, a_chain)
        damage_line_2_fairlead_top_chain = compute_damage(stress_line_2_fairlead_top_chain, m_chain, a_chain)
        damage_line_3_fairlead_top_chain = compute_damage(stress_line_3_fairlead_top_chain, m_chain, a_chain)
        damage_line_4_fairlead_top_chain = compute_damage(stress_line_4_fairlead_top_chain, m_chain, a_chain)
        
        damage_line_1_fairlead_rope = compute_damage(df_out["FAIRTEN2_[N]"][transient_time_index_out:] / MBS_rope, m_rope, a_rope)
        damage_line_2_fairlead_rope = compute_damage(df_out["FAIRTEN5_[N]"][transient_time_index_out:] / MBS_rope, m_rope, a_rope)
        damage_line_3_fairlead_rope = compute_damage(df_out["FAIRTEN8_[N]"][transient_time_index_out:] / MBS_rope, m_rope, a_rope)
        damage_line_4_fairlead_rope = compute_damage(df_out["FAIRTEN11_[N]"][transient_time_index_out:] / MBS_rope, m_rope, a_rope)
        
        damage_line_1_fairlead_bottom_chain = compute_damage(stress_line_1_fairlead_bottom_chain, m_chain, a_chain)
        damage_line_2_fairlead_bottom_chain = compute_damage(stress_line_2_fairlead_bottom_chain, m_chain, a_chain)
        damage_line_3_fairlead_bottom_chain = compute_damage(stress_line_3_fairlead_bottom_chain, m_chain, a_chain)
        damage_line_4_fairlead_bottom_chain = compute_damage(stress_line_4_fairlead_bottom_chain, m_chain, a_chain)
        
        # --- DEL -------
        
        DEL_line_1_fairlead_top_chain = compute_DEL(stress_line_1_fairlead_top_chain, m_chain)
        DEL_line_2_fairlead_top_chain = compute_DEL(stress_line_2_fairlead_top_chain, m_chain)
        DEL_line_3_fairlead_top_chain = compute_DEL(stress_line_3_fairlead_top_chain, m_chain)
        DEL_line_4_fairlead_top_chain = compute_DEL(stress_line_4_fairlead_top_chain, m_chain)
        
        DEL_line_1_fairlead_rope = compute_DEL(df_out["FAIRTEN2_[N]"][transient_time_index_out:] / MBS_rope, m_rope)
        DEL_line_2_fairlead_rope = compute_DEL(df_out["FAIRTEN5_[N]"][transient_time_index_out:] / MBS_rope, m_rope)
        DEL_line_3_fairlead_rope = compute_DEL(df_out["FAIRTEN8_[N]"][transient_time_index_out:] / MBS_rope, m_rope)
        DEL_line_4_fairlead_rope = compute_DEL(df_out["FAIRTEN11_[N]"][transient_time_index_out:] / MBS_rope, m_rope)
        
        DEL_line_1_fairlead_bottom_chain = compute_DEL(stress_line_1_fairlead_bottom_chain, m_chain)
        DEL_line_2_fairlead_bottom_chain = compute_DEL(stress_line_2_fairlead_bottom_chain, m_chain)
        DEL_line_3_fairlead_bottom_chain = compute_DEL(stress_line_3_fairlead_bottom_chain, m_chain)
        DEL_line_4_fairlead_bottom_chain = compute_DEL(stress_line_4_fairlead_bottom_chain, m_chain)

        # Save results to CSV
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                wind_speed,
                significant_wave_height,
                peak_period,
                0.0, 
                0.0, 
                l_rope,
                d_chain,
                d_rope,
                radius,
                z_fairlead,  
                mean_pitch_angles,
                max_pitch_angles,
                mean_roll_angles,
                max_roll_angles,
                max_vertical_accs,
                max_horizontal_accs,
                max_yaw_ptfms,
                floater_exursions_surge,
                floater_exursions_sway,
                max_tensions_line_1_fairlead_top_chain,
                max_tensions_line_2_fairlead_top_chain,
                max_tensions_line_3_fairlead_top_chain,
                max_tensions_line_4_fairlead_top_chain,
                max_tensions_line_1_fairlead_rope,
                max_tensions_line_2_fairlead_rope,
                max_tensions_line_3_fairlead_rope,
                max_tensions_line_4_fairlead_rope,
                max_tensions_line_1_fairlead_bottom_chain,
                max_tensions_line_2_fairlead_bottom_chain,
                max_tensions_line_3_fairlead_bottom_chain,
                max_tensions_line_4_fairlead_bottom_chain,
                mean_tensions_line_1_fairlead_top_chain,
                mean_tensions_line_2_fairlead_top_chain,
                mean_tensions_line_3_fairlead_top_chain,
                mean_tensions_line_4_fairlead_top_chain,
                mean_tensions_line_1_fairlead_rope,
                mean_tensions_line_2_fairlead_rope,
                mean_tensions_line_3_fairlead_rope,
                mean_tensions_line_4_fairlead_rope,
                mean_tensions_line_1_fairlead_bottom_chain,
                mean_tensions_line_2_fairlead_bottom_chain,
                mean_tensions_line_3_fairlead_bottom_chain,
                mean_tensions_line_4_fairlead_bottom_chain,
                percentile_tensions_line_1_fairlead_top_chain,
                percentile_tensions_line_2_fairlead_top_chain,
                percentile_tensions_line_3_fairlead_top_chain,
                percentile_tensions_line_4_fairlead_top_chain,
                percentile_tensions_line_1_fairlead_rope,
                percentile_tensions_line_2_fairlead_rope,
                percentile_tensions_line_3_fairlead_rope,
                percentile_tensions_line_4_fairlead_rope,
                percentile_tensions_line_1_fairlead_bottom_chain,
                percentile_tensions_line_2_fairlead_bottom_chain,
                percentile_tensions_line_3_fairlead_bottom_chain,
                percentile_tensions_line_4_fairlead_bottom_chain,
                damage_line_1_fairlead_top_chain,
                damage_line_2_fairlead_top_chain,
                damage_line_3_fairlead_top_chain,
                damage_line_4_fairlead_top_chain,
                damage_line_1_fairlead_rope,
                damage_line_2_fairlead_rope,
                damage_line_3_fairlead_rope,
                damage_line_4_fairlead_rope,
                damage_line_1_fairlead_bottom_chain,
                damage_line_2_fairlead_bottom_chain,
                damage_line_3_fairlead_bottom_chain,
                damage_line_4_fairlead_bottom_chain,
                DEL_line_1_fairlead_top_chain,
                DEL_line_2_fairlead_top_chain,
                DEL_line_3_fairlead_top_chain,
                DEL_line_4_fairlead_top_chain,
                DEL_line_1_fairlead_rope,
                DEL_line_2_fairlead_rope,
                DEL_line_3_fairlead_rope,
                DEL_line_4_fairlead_rope,
                DEL_line_1_fairlead_bottom_chain,
                DEL_line_2_fairlead_bottom_chain,
                DEL_line_3_fairlead_bottom_chain,
                DEL_line_4_fairlead_bottom_chain
            ])

    
if __name__ == "__main__":

    for i in range(119, 1001):
        sim_folder = f"Design_{i}"
        sim_name = f"sample_{i}"
        current_folder = 'Saved_Outputs'
        seeds_folders = ['Seed_1', 'Seed_2', 'Seed_3', 'Seed_4', 'Seed_5', 'Seed_6']
        file_settings = "../Inputs_Definition/Settings.yaml"
        input_file = "../Inputs_Definition/DOE_Design_Environmental_Definitions.yaml"

        analyze_simulation_results(file_settings, sim_folder, current_folder, seeds_folders, input_file, sim_name)


