#%% Import packages
import os
import sys
import time
import yaml
import numpy as np
#from openfast_toolbox.io.fast_input_file import FASTInputFile
from hipersim.turbgen.generate_field import generate_field

#%% Mann Turbulence Model Function
def mann_model(Umean, Nx, Ny, Nz, dx, dy, dz, Tsim, base_name, seed_no):
    """
    Generates a turbulent wind field using the Mann model.
    
    Documentation: 
    https://gitlab.windenergy.dtu.dk/HiperSim/hipersim/-/blob/master/examples/Turbgen/Turbgen_Manual_0.0.24.ipynb

    Parameters:
    - Umean: Mean wind speed
    - Nx, Ny, Nz: Grid dimensions
    - dx, dy, dz: Spatial resolutions
    - Tsim: Simulation time
    - base_name: Base name for output files
    - seed_no: Random seed for turbulence generation
    """
    L = 29.4  # Turbulence length scale
    Gamma = 3.9  # Shear parameter
    alpha_epsilon = 1  # Energy dissipation parameter
    high_freq_comp = 1  # High-frequency component flag

    generate_field(
        base_name, alpha_epsilon, L, Gamma, seed_no, Nx, Ny, Nz, dx, dy, dz, high_freq_comp, SaveToFile=True
    )

#%% Main Wind Generation Function
def generate_wind(sim_folder):
    """
    Generates wind files for a given simulation folder.
    
    Parameters:
    - sim_folder: Name of the simulation folder.
    """
    settings_file = "../../../../../../Inputs_Definition/Settings.yaml"

    # Load simulation settings from YAML
    if not os.path.exists(settings_file):
        raise FileNotFoundError(f"Settings file not found: {settings_file}")
    
    with open(settings_file, "r") as file:
        data = yaml.safe_load(file)

    sim_time = data["time"]["sim_time"]
    filename = f'IEA-15-240-RWT_InflowFile_{sim_folder}.dat'

    # Read InflowFile
    try:
        f = FASTInputFile(filename)
    except Exception as e:
        raise RuntimeError(f"Error reading FAST input file: {filename}") from e

    # Extract parameters from FAST file
    Nx, Ny, Nz = f['nx'], f['ny'], f['nz']
    dx, dy, dz = f['dx'], f['dy'], f['dz']
    z_hub, U_hub = f['RefHt_Hawc'], f['URef']

    # Set wind turbulence base name and random seed
    base_name = 'Mann_turb'
    wind_seed = np.random.randint(1, 2147483648)  # Random seed for wind generation

    # Update FAST input file with generated wind file names
    f['FileName_u'] = f"{base_name}_{wind_seed}_u.bin"
    f['FileName_v'] = f"{base_name}_{wind_seed}_v.bin"
    f['FileName_w'] = f"{base_name}_{wind_seed}_w.bin"
    f['RotorApexOffsetPos'] = "0.0 0.0 0.0"  # Explicitly set to avoid potential format issues

    f.write(filename)

    # Generate wind turbulence field
    start_time = time.time()
    mann_model(U_hub, Nx, Ny, Nz, dx, dy, dz, sim_time, base_name, wind_seed)
    print(f"--- Wind generation completed in {time.time() - start_time:.2f} seconds ---")

#%% Command-Line Execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python wind_generation.py <sim_folder>")
        sys.exit(1)

    try:
        generate_wind(sys.argv[1])
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)
