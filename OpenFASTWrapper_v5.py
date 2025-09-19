import numpy as np
import os
import shutil
import yaml
import math
import pandas as pd
import openfast_toolbox.case_generation.case_gen as case_gen
import openfast_toolbox.case_generation.runner as runner
import openfast_toolbox.postpro as postpro
from openfast_toolbox.io.fast_input_file import FASTInputFile
from OpenFASTWrapper_design_v3 import update_mooring_system


class SlurmBatchWrapper:
    def __init__(self, ref_dir, main_file, work_dir, config_env_design_file, config_sim_file, IC_file,
                 IC_file_rotor, working_conda_environment, script_turb_wind_gen, script_postprocessing, job_name="slurm_job", 
                 partition="general", time="24:00:00", nodes=1, ntasks_per_node=1, array_jobs=False, 
                 max_parallel_jobs=None, email=None, stdOutToFile=False):
        """
        Initialize the wrapper with user-defined input files and SLURM job settings.

        Args:
            ref_dir (str): Path to the reference OpenFAST model directory.
            main_file (str): Name of the main .fst file used as a template.
            work_dir (str): Directory where output files will be created.
            config_env_file (str): Path to the YAML file containing environmental conditions.
            config_sim_file (str): Path to the YAML file containing simulation settings.
            IC_file (str): Path to the CSV file containing all the initial conditions for different wind speeds and headings.
            IC_file_rotor (str): Path to the CSV file containing all the initial conditions of the rotor for different wind speeds and headings.
            job_name (str, optional): SLURM job name.
            partition (str, optional): SLURM partition to use.
            time (str, optional): Maximum execution time.
            nodes (int, optional): Number of compute nodes.
            ntasks_per_node (int, optional): Number of tasks per node.
            array_jobs (bool, optional): Whether to use SLURM job arrays.
            max_parallel_jobs (int, optional): Max parallel jobs in a job array.
            email (str, optional): Email for job notifications.
            stdOutToFile (bool, optional): Redirect stdout to a file.
        """
        self.ref_dir = ref_dir
        self.main_file = main_file
        self.work_dir = work_dir
        self.config_env_file = config_env_design_file
        self.config_sim_file = config_sim_file
        self.IC_file = IC_file
        self.IC_file_rotor = IC_file_rotor

        # SLURM parameters
        self.job_name = job_name
        self.partition = partition
        self.time = time
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.array_jobs = array_jobs
        self.max_parallel_jobs = max_parallel_jobs
        self.email = email
        self.stdOutToFile = stdOutToFile
        self.working_conda_environment = working_conda_environment
        self.script_turb_wind_gen = script_turb_wind_gen
        self.script_postprocessing = script_postprocessing

        # Default simulation settings
        self.base_dict = {}

        # Load YAML files
        self.sim_settings = self._load_yaml(self.config_sim_file)
        self.sim_designs = self._load_yaml(self.config_env_file)
        self.IC = pd.read_csv(self.IC_file)
        self.IC_rotor = pd.read_csv(self.IC_file_rotor)

        # Extracted parameters
        self.PARAMS = []

        # Initialize parameters
        self._initialize_params()

    def _load_yaml(self, file_path):
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

    def _initialize_params(self):
        """Extracts and initializes necessary parameters from the config files."""
        self.sim_time = self.sim_settings["time"]["sim_time"]
        self.rotor_diameter = self.sim_settings["turbine"]["rotor_diameter"]
        self.hub_height = self.sim_settings["turbine"]["hub_height"]
        self.reference_height = self.sim_settings["wind"]["reference_height"]
        self.alpha = self.sim_settings["wind"]["alpha"]
        self.Ny = self.sim_settings["wind"]["Ny"]
        self.Nz = self.sim_settings["wind"]["Nz"]
        self.dx = self.sim_settings["wind"]["dx"]
        self.seeds_number = self.sim_settings["wind"]["seeds_number"]
        self.depth = self.sim_settings["mooring_layout"]["depth"]
        self.l_top_chain = self.sim_settings["mooring_layout"]["l_top_chain"]
        self.l_bottom_chain = self.sim_settings["mooring_layout"]["l_bottom_chain"]
        self.x_fairlead = self.sim_settings["mooring_layout"]["x_fairlead_line_1"]
        self.y_fairlead = self.sim_settings["mooring_layout"]["y_fairlead_line_1"]
        self.z_fairlead = self.sim_settings["mooring_layout"]["z_fairlead_line_1"]
        self.x_anchor = self.sim_settings["mooring_layout"]["x_anchor_line_1"]
        self.y_anchor = self.sim_settings["mooring_layout"]["y_anchor_line_1"]
        self.z_anchor = self.sim_settings["mooring_layout"]["z_anchor_line_1"]
        self.z_in_line_buoy_wr_seafloor = self.sim_settings["mooring_layout"]["z_in_line_buoy_wr_seafloor"]
        

    def _update_mooring_layout(self): 
        """Extracts and derive the updated mooring layout and parameters from the config files."""
        for exp in self.sim_designs.get("experiments", []):
            doe_number = exp["experiment_id"]
            d_chain = exp["d_chain"]
            d_rope = exp["d_rope"]*d_chain
            radius = exp["radius"]
            l_rope = exp["L_rope"]* radius * 1.01956
            z_fairlead = exp["z_fairlead"]
            print('d_rope', d_rope, 'd_chain', d_chain, 'l_rope', l_rope, "z_fairlead", z_fairlead,  "radius", radius, 'doe_number', doe_number)
            update_mooring_system(d_rope, d_chain , l_rope, z_fairlead, radius,
                          self.config_sim_file, "../Inputs_Definition/Mooring_Designs_Input_Files/",
                          doe_number)
            
            if not os.path.exists(self.work_dir + 'Design_' + str(doe_number)):
                os.makedirs(self.work_dir + 'Design_' + str(doe_number))  # Creates the folder (and parent directories if needed)
    
    def _update_mooring_file(self, doe_number, input_folder, output_folder): 
        """
        Copies a specified file from the input folder into each subfolder inside the output folder.

        :param input_folder: Path to the folder containing the file to be copied.
        :param output_folder: Path to the folder containing subfolders where the file should be copied.
        :param file_name: Name of the file to copy.
        """
        file_name = "IEA-15-240-RWT-Nautilus_MoorDyn_synthetic" + "_design_" + str(doe_number) + ".dat"
        input_file_path = os.path.join(input_folder, file_name)
        print('input_file_path', input_file_path)

        # Check if the input file exists
        if not os.path.exists(input_file_path):
            print(f"Error: The file '{file_name}' does not exist in '{input_folder}'.")
            return
        
        # Loop through all items in the output folder
        for folder in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder)

            # Ensure it's a directory
            if os.path.isdir(folder_path):
                destination_file_path = os.path.join(folder_path, file_name)
                print('destination_file_path', destination_file_path)
                
                # Copy the file to the subfolder
                shutil.copy2(input_file_path, './' + folder_path)
                print(f"Copied {file_name} to {folder_path}")  
                
                # Update .fst main file
                filename = os.path.join(folder_path, folder + '.fst')
                f = FASTInputFile(filename)
                f['MooringFile'] = file_name
                f.write(filename)
                print(f"Updated {filename}")
    
    def _generate_simulation_cases(self, exp):
        """Generates simulation parameters for each case."""
        print("Generating simulation cases...")
        
        sim = exp
        
        p = self.base_dict.copy()
        
        #Main File SImulations Parameters 
        p["TMax"] = self.sim_time

        # Compute Wind Speed at Hub Height
        U_hub = ((self.hub_height / self.reference_height) ** self.alpha) * sim["hub_wind_speed"]
        wave_seed = np.random.randint(-2147483648,2147483648)#-2147483648 to 2147483647
        
        # Interpolate on ICs
        if (U_hub < self.sim_settings["turbine"]["cut_in_speed"]):
            rpm_init = 0
            pitch_init = 0
        
        elif (U_hub >= self.sim_settings["turbine"]["cut_in_speed"]) and (U_hub < self.sim_settings["turbine"]["cut_out_speed"]):
            #rpm_init = np.interp(U_hub, self.IC_rotor['Wind [m/s]'].values, self.IC_rotor['Rotor Speed [rpm]'].values)       
            rpm_init = 3.0                               
            #pitch_init = np.interp(U_hub, self.IC_rotor['Wind [m/s]'].values, self.IC_rotor['Pitch [deg]'].values)
            pitch_init = 45.0                 
        
        elif (U_hub >= self.sim_settings["turbine"]["cut_out_speed"]):
            rpm_init = 0
            pitch_init = 90
    
        #From the wind direction, wind speed, wave height and wave period 
        # Interpolate over the computed ICs and modify the initial position and initial rotor speed of the model 
        Initial_RotSpeed = rpm_init
        Initial_PitchAngle_Blade = pitch_init
        
        #self.IC['Distance'] = np.sqrt((self.IC['Wind Direction (deg)'] - sim["wind_direction"]) ** 2 + 
        #                (self.IC['Wind Speed (m/s)'] - U_hub) ** 2)
        
        Initial_PtfmSurge = 0.0 # self.IC.loc[self.IC['Distance'].idxmin(), 'Surge Offset (m)']
        Initial_PtfmSway = 0.0 # self.IC.loc[self.IC['Distance'].idxmin(), 'Sway Offset (m)']
        Initial_PtfmHeave = 0.0 # self.IC.loc[self.IC['Distance'].idxmin(), 'Heave Offset (m)']
        Initial_PtfmRoll = 0.0 # self.IC.loc[self.IC['Distance'].idxmin(), 'Roll Offset (deg)']
        Initial_PtfmPitch = 0.0 # self.IC.loc[self.IC['Distance'].idxmin(), 'Pitch Offset (deg)']
        Initial_PtfmYaw = 0.0 # self.IC.loc[self.IC['Distance'].idxmin(), 'Yaw Offset (deg)']
        
        
        if (U_hub < self.sim_settings["turbine"]["cut_in_speed"]) or (U_hub > self.sim_settings['turbine']['cut_out_speed']):
            # Park the turbine
            p["CompServo"] = 0
            p["ServoFile|PCMode"] = 0
            p["ServoFile|VSContrl"] = 0
            p["EDFile|GenDOF"] = 'False'
            p["EDFile|BlPitch(1)"] = Initial_PitchAngle_Blade
            p["EDFile|BlPitch(2)"] = Initial_PitchAngle_Blade
            p["EDFile|BlPitch(3)"] = Initial_PitchAngle_Blade
            p["EDFile|RotSpeed"] = Initial_RotSpeed
            p["EDFile|NacYaw"] = 0.0
        else:
            p["CompServo"] = 1
            p["ServoFile|PCMode"] = 5
            p["ServoFile|VSContrl"] = 5
            p["EDFile|GenDOF"] = 'True'
            p["EDFile|BlPitch(1)"] = Initial_PitchAngle_Blade
            p["EDFile|BlPitch(2)"] = Initial_PitchAngle_Blade
            p["EDFile|BlPitch(3)"] = Initial_PitchAngle_Blade
            p["EDFile|RotSpeed"] = Initial_RotSpeed
            p["EDFile|NacYaw"] = 0.0
        
        p["EDFile|PtfmSurge"] = Initial_PtfmSurge
        p["EDFile|PtfmSway"] = Initial_PtfmSway
        p["EDFile|PtfmHeave"] = Initial_PtfmHeave
        p["EDFile|PtfmRoll"] = Initial_PtfmRoll
        p["EDFile|PtfmPitch"] = Initial_PtfmPitch
        p["EDFile|PtfmYaw"] = Initial_PtfmYaw
        p["EDFile|YawDOF"] = 'False'                           

        # Assign Simulation Parameters
        p.update({
            "InflowFile|WindType": 5,
            "InflowFile|PropagationDir": 0.0,
            "InflowFile|nx": 2 ** (math.ceil(math.log2(U_hub * self.sim_time / self.dx))),
            "InflowFile|ny": self.Ny,
            "InflowFile|nz": self.Nz,
            "InflowFile|dx": self.dx,
            "InflowFile|dy": 300 / self.Ny,
            "InflowFile|dz": 300 / self.Nz,
            "InflowFile|Uref": U_hub,
            "InflowFile|WindProfile": 2,
            "InflowFile|PLExp_Hawc": self.alpha,
            "InflowFile|RefHt_Hawc": self.hub_height,
            "InflowFile|RotorApexOffsetPos": "0.0 0.0 0.0",
            "__name__": "sample_" + str(sim["experiment_id"]),  # Define name for .fst file
            "SeaStFile|WaveMod": 2,
            "SeaStFile|WaveHs": sim["significant_wave_height"],
            "SeaStFile|WaveTp": sim["peak_period"],
            "SeaStFile|WaveDir": 0.0,
            "SeaStFile|CurrMod": 1,
            "SeaStFile|CurrNSV0": 0.0,
            "SeaStFile|CurrNSRef": 3.0,
            "SeaStFile|CurrNSDir": 0.0,
            "SeaStFile|WaveSeed(1)": wave_seed,
            "SeaStFile|WaveTMax": self.sim_time,
        })

        # Define JONSWAP Peak Enhancement Factor
        Tp_sqrt_Hs = sim["peak_period"] / np.sqrt(sim["significant_wave_height"])
        if Tp_sqrt_Hs <= 3.6:
            p["SeaStFile|WavePkShp"] = 5
        elif Tp_sqrt_Hs > 5:
            p["SeaStFile|WavePkShp"] = 1
        else:
            p["SeaStFile|WavePkShp"] = np.exp(5.75 - 1.15 * Tp_sqrt_Hs)

        self.PARAMS.append(p)

    def _create_slurm_batch(self, fast_files, num_samples):
        """Generates a SLURM batch script for parallel execution."""
        print("Creating SLURM batch script...")

        batchfile = os.path.join(self.work_dir, "_RUN_ALL_ARRAY.job")

        with open(batchfile, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --partition={self.partition}
#SBATCH --time={self.time}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.ntasks_per_node}
""")

            if self.email:
                f.write(f"""#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={self.email}
""")

            if self.array_jobs:
                f.write(f"""#SBATCH --array=1-{num_samples}""" + (f"%{self.max_parallel_jobs}" if self.max_parallel_jobs else "") + "\n")

                f.write(f"""module use -a /apps/external/sapps/modules/app
module load openfast/4.0.0
""")

            if self.array_jobs:
                f.write("SIM_FOLDER=sample${SLURM_ARRAY_TASK_ID}\n")
                f.write("for CASE_DIR in */; do\n")
                f.write("""   echo "Entering $CASE_DIR..."
    cd $CASE_DIR/Seed_1 || { echo "Failed to enter $CASE_DIR"; exit 1; }\n""")
                f.write("   cd $SIM_FOLDER/IEA-15-240-RWT|| { echo \"Failed to enter $SIM_FOLDER\"; exit 1; }\n")
                f.write(f"""    {self.working_conda_environment} {self.script_turb_wind_gen} $SIM_FOLDER\n""")
                f.write("   cd .. || { echo \"Failed to enter $SIM_FOLDER\"; exit 1; }\n")
                f.write(f"  openfast $SIM_FOLDER.fst\n")
                f.write("  cd ../..\n")
                f.write("done\n")
                #f.write(f"  {self.working_conda_environment} {self.script_postprocessing} $SIM_FOLDER\n")
                f.write("""echo "All simulations in $SIM_FOLDER completed."\n""")
            else:
                for ff in fast_files:
                    ff_rel = os.path.relpath(ff, self.work_dir)
                    f.write(f"""module use -a /apps/external/sapps/modules/app
module load openfast/4.0.0
""")
                    cmd = f"openfast {ff_rel}"
                    if self.stdOutToFile:
                        stdout = os.path.splitext(ff_rel)[0] + '.stdout'
                        cmd += f' > {stdout}'
                    f.write(f"{cmd}\n")

        print(f"SLURM batch script created: {batchfile}")

    def run(self):
        """Runs the full workflow: Load params, generate cases, and create SLURM batch."""
        self._update_mooring_layout()
        for exp in self.sim_designs.get("experiments", []):
            for i in range(self.seeds_number): 
                self._generate_simulation_cases(exp)

                print("Generating FAST input files...")
                fast_files = case_gen.templateReplace(
                    self.PARAMS, self.ref_dir, outputDir=self.work_dir + 'Design_' + str(exp["experiment_id"]) + '/' + 'Seed_'+str(i+1)+'/',
                    removeRefSubFiles=True, main_file=self.main_file, oneSimPerDir=True
                )
                
                # Extracted parameters
                self.PARAMS = []

                print(f"Generated {len(fast_files)} FAST input files.")
                print(self.work_dir + 'Design_' + str(exp["experiment_id"]) + '/' + 'Seed_'+str(i+1)+'/')
                self._update_mooring_file(exp["experiment_id"], 
                                          "../Inputs_Definition/Mooring_Designs_Input_Files/", 
                                          self.work_dir + 'Design_' + str(exp["experiment_id"]) + '/' + 'Seed_'+str(i+1)+'/')
            num_samples = len(fast_files)//self.seeds_number
            self._create_slurm_batch(fast_files, num_samples)
            print("SLURM batch script successfully created!")


# === Run the script ===
if __name__ == "__main__":
    wrapper = SlurmBatchWrapper(
        ref_dir="../OpenFAST_Base_Case_Model/",
        main_file="IEA-15-240-RWT-Nautilus.fst",
        work_dir="Nautilus_15MW_Parametric/",
        config_env_design_file="../Inputs_Definition/DOE_Design_Environmental_Definitions_subset.yaml",
        config_sim_file="../Inputs_Definition/Settings.yaml",
        IC_file ="../Inputs_Definition/Initial_Conditions.csv",
        IC_file_rotor ="../Inputs_Definition/Rotor_Performances.csv",
        email="ajmlu@dtu.dk",
        array_jobs=True,
        max_parallel_jobs=10,
        ntasks_per_node=32,
        working_conda_environment = '/work/users/ajmlu/miniconda3/envs/openfast_env_v1/bin/python', 
        script_turb_wind_gen = '/home/ajmlu/Python_Framework_Design_Constraints/02_Design_Exploration_Surrogate_Model/Cases/wind_generation.py',
        script_postprocessing = '/home/ajmlu/Python_Framework_Design_Constraints/02_Design_Exploration_Surrogate_Model/Cases/PostProcessing.py',
        partition = 'windq',
        time = "24:00:00"
    )

    wrapper.run()
