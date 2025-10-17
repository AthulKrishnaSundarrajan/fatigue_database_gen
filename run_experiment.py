import numpy as np
import os
import shutil
import yaml
import math
import pandas as pd
from rosco import discon_lib_path
from copy import copy
#import openfast_toolbox.case_generation.case_gen as case_gen
#import openfast_toolbox.postpro as postpro
#from openfast_toolbox.io.fast_input_file import FASTInputFile
from Cases.OpenFASTWrapper_design_v3 import update_mooring_system
from load_yaml import load_yaml

from openfast_io.FAST_reader import InputReader_OpenFAST
from openfast_io.FAST_writer import InputWriter_OpenFAST

from wind_generation import mann_model
import time as timer
from run_openfast import run_openfast

class run_experiment():

    '''
    Class that holds all the necessary information
    
    '''

    def __init__(self, of_dir,main_file, work_dir, config_env_design_file, config_sim_file):
        """
        Initialize the wrapper with user-defined input files and SLURM job settings.

        Args:
            ref_dir (str): Path to the reference OpenFAST model directory.
            main_file (str): Name of the main .fst file used as a template.
            work_dir (str): Directory where output files will be created.
            config_env_file (str): Path to the YAML file containing environmental conditions.
            config_sim_file (str): Path to the YAML file containing simulation settings.
        """
        #self.ref_dir = ref_dir
        self.of_dir = of_dir
        self.main_file = main_file
        self.work_dir = work_dir
        self.config_env_file = config_env_design_file
        self.config_sim_file = config_sim_file


        # Default simulation settings
        self.base_dict = {}

        # Load YAML files
        self.sim_settings = load_yaml(self.config_sim_file)
        self.sim_designs = load_yaml(self.config_env_file)
        
        #self.IC = pd.read_csv(self.IC_file)
        #self.IC_rotor = pd.read_csv(self.IC_file_rotor)

        # Extracted parameters
        self.PARAMS = []
        self.fst_files = []

        # Initialize parameters
        self._initialize_params()
        

        self.wind_dirs = []

        for exp in self.sim_designs.get("experiments", []):
            for i in range(self.seeds_number): 
                

                fst_dir = self.of_dir
                fst_file = self.main_file

                of_reader = InputReader_OpenFAST()
                of_reader.FAST_InputFile = fst_file
                of_reader.FAST_directory = fst_dir

                of_reader.read_MainInput()
                
                of_reader.read_ElastoDyn(fst_dir + os.sep +of_reader.fst_vt['Fst']['EDFile'])
                of_reader.read_ElastoDynBlade(fst_dir + os.sep +of_reader.fst_vt['ElastoDyn']['BldFile1'])
                of_reader.read_ElastoDynTower(fst_dir + os.sep +of_reader.fst_vt['ElastoDyn']['TwrFile'])
                of_reader.read_AeroDyn()
                of_reader.read_ServoDyn()
                of_reader.read_SeaState(fst_dir + os.sep +of_reader.fst_vt['Fst']['SeaStFile'])
                of_reader.read_MoorDyn(fst_dir + os.sep +of_reader.fst_vt['Fst']['MooringFile'])
                of_reader.read_HydroDyn(fst_dir + os.sep +of_reader.fst_vt['Fst']['HydroFile'])
                of_reader.read_InflowWind()
                of_reader.read_DISCON_in()

                of_reader.fst_vt['ElastoDynBlade'] = of_reader.fst_vt['ElastoDynBlade'][0]

                fst_vt = of_reader.fst_vt

                fst_copy = copy(fst_vt)

                outputDir=self.work_dir + 'Design_' + str(exp["experiment_id"]) + os.sep +'seed_no_'+str(i)

                self._generate_simulation_cases(exp,fst_copy,outputDir,self.main_file)

                

        self._update_mooring_layout()

        breakpoint()


    def _generate_simulation_cases(self, exp,fst_vt,output_dir,main_file):
        """Generates simulation parameters for each case."""
        print("Generating simulation cases...")
        
        sim = exp
        
        p = self.base_dict.copy()

        of_writer = InputWriter_OpenFAST()
        of_writer.FAST_InputFile = main_file
        of_writer.FAST_runDirectory = output_dir
        of_writer.FAST_namingOut = 'IEA-15-240-RWT-Nautilus'

        self.fst_files.append(output_dir + os.sep+main_file)
        
        #Main File SImulations Parameters 
        fst_vt['Fst']['TMax'] = self.sim_time
        #p["TMax"] = self.sim_time

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
            fst_vt['Fst']["CompServo"] = 0
            fst_vt['ServoDyn']['PCMode'] = 0
            fst_vt['ServoDyn']['VSContrl'] = 0
            fst_vt['ElastoDyn']['GenDOF'] = False
            fst_vt['ElastoDyn']['BlPitch(1)'] = Initial_PitchAngle_Blade
            fst_vt['ElastoDyn']['BlPitch(2)'] = Initial_PitchAngle_Blade
            fst_vt['ElastoDyn']['BlPitch(3)'] = Initial_PitchAngle_Blade
            fst_vt['ElastoDyn']['RotSpeed'] = Initial_RotSpeed
            
        else:
            fst_vt['Fst']["CompServo"] = 1
            fst_vt['ServoDyn']['PCMode'] = 5
            fst_vt['ServoDyn']['VSContrl'] = 5
            fst_vt['ElastoDyn']['GenDOF'] = True
            fst_vt['ElastoDyn']['BlPitch(1)'] = Initial_PitchAngle_Blade
            fst_vt['ElastoDyn']['BlPitch(2)'] = Initial_PitchAngle_Blade
            fst_vt['ElastoDyn']['BlPitch(3)'] = Initial_PitchAngle_Blade
            fst_vt['ElastoDyn']['RotSpeed'] = Initial_RotSpeed
        
        fst_vt['ServoDyn']['DLL_FileName'] = discon_lib_path
        fst_vt['ElastoDyn']['PtfmSurge'] = Initial_PtfmSurge
        fst_vt['ElastoDyn']['PtfmSway'] = Initial_PtfmSway
        fst_vt['ElastoDyn']['PtfmHeave'] = Initial_PtfmHeave
        fst_vt['ElastoDyn']['PtfmRoll'] = Initial_PtfmRoll
        fst_vt['ElastoDyn']['PtfmPitch'] = Initial_PtfmPitch
        fst_vt['ElastoDyn']['PtfmYaw'] = Initial_PtfmYaw
        fst_vt['ElastoDyn']['NacYaw'] = 0.0
        fst_vt['ElastoDyn']['YawDOF'] = False                           

        # Assign Simulation Parameters
        fst_vt['InflowWind']['WindType'] = 5
        fst_vt['InflowWind']['PropagationDir'] = 0
        fst_vt['InflowWind']['nx'] = 2 ** (math.ceil(math.log2(U_hub * self.sim_time / self.dx)))
        fst_vt['InflowWind']['ny'] = self.Ny
        fst_vt['InflowWind']['nz'] = self.Nz
        fst_vt['InflowWind']['dx'] = self.dx
        fst_vt['InflowWind']['dy'] = 300/self.Ny
        fst_vt['InflowWind']['dz'] = 300/self.Nz
        fst_vt['InflowWind']['Uref'] = U_hub
        fst_vt['InflowWind']['WindProfile'] = 2
        fst_vt['InflowWind']['PLExp_Hawc'] = self.alpha
        fst_vt['InflowWind']['RefHt_Hawc']=self.hub_height
        wind_seed = np.random.randint(1, 2147483648)
        fst_vt['InflowWind']['FileName_u'] = output_dir + os.sep+ 'wind'+os.sep +'Mann_turb_'+str(wind_seed)+'_u.bin'
        fst_vt['InflowWind']['FileName_v'] = output_dir + os.sep+ 'wind' +os.sep +'Mann_turb_'+str(wind_seed)+'_v.bin'
        fst_vt['InflowWind']['FileName_w'] = output_dir + os.sep+ 'wind' +os.sep + 'Mann_turb_'+str(wind_seed)+'_w.bin'

        fst_vt['SeaState']['WaveMod'] = 2
        fst_vt['SeaState']['WaveHs'] = sim["significant_wave_height"]
        fst_vt['SeaState']['WaveTp'] = sim["peak_period"]
        fst_vt['SeaState']['WaveDir'] = 0
        fst_vt['SeaState']['CurrMod'] = 1
        fst_vt['SeaState']['CurrNSRef'] = 3.0
        fst_vt['SeaState']['WaveSeed(1)'] = wave_seed
        fst_vt['SeaState']['WaveTMax'] = self.sim_time
        #fst_vt['DISCON_in']['VS_FBP'] = 0
        #fst_vt['DISCON_in']['SU_Mode'] = 0
        #fst_vt['DISCON_in']['SD_Mode'] = 0
        #fst_vt['DISCON_in']['F_VSRefSpdCornerFreq'] = 0
        #fst_vt['DISCON_in']['VS_FBP_n'] = 0
        #fst_vt['DISCON_in']["VS_FBP_Omega"] = [0]
        #fst_vt['DISCON_in']["VS_FBP_Tau"] = [0]
        #fst_vt['DISCON_in']['VS_FBP_U'] = 10

        # Define JONSWAP Peak Enhancement Factor
        Tp_sqrt_Hs = sim["peak_period"] / np.sqrt(sim["significant_wave_height"])
        if Tp_sqrt_Hs <= 3.6:
            fst_vt['SeaState']['WavePkShp'] = 5
        elif Tp_sqrt_Hs > 5:
            fst_vt['SeaState']['WavePkShp'] = 1
        else:
            fst_vt['SeaState']['WavePkShp'] = np.exp(5.75 - 1.15 * Tp_sqrt_Hs)

        base_name = 'Mann_turb'
        #wind_seed = np.random.randint(1, 2147483648)  # Random seed for wind generation

        if os.path.exists(output_dir + os.sep+ 'wind'):
            os.rmdir(output_dir + os.sep+ 'wind')
        
        os.mkdir(output_dir + os.sep+ 'wind')
        self.wind_dirs.append(output_dir + os.sep+ 'wind')

        # Generate wind turbulence field
        start_time = timer.time()
        mann_model(U_hub, fst_vt['InflowWind']['nx'], fst_vt['InflowWind']['ny'], fst_vt['InflowWind']['nz'], fst_vt['InflowWind']['dx'], fst_vt['InflowWind']['dy'], fst_vt['InflowWind']['dz'], self.sim_time,
         output_dir + os.sep+ 'wind' +os.sep +base_name, wind_seed)
        print(f"--- Wind generation completed in {timer.time() - start_time:.2f} seconds ---")

        of_writer.fst_vt = fst_vt

        of_writer.execute()
        

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

            if not os.path.exists(self.work_dir + 'Design_' + str(doe_number)):
                os.makedirs(self.work_dir + 'Design_' + str(doe_number))  # Creates the folder (and parent directories if needed)

            print('d_rope', d_rope, 'd_chain', d_chain, 'l_rope', l_rope, "z_fairlead", z_fairlead,  "radius", radius, 'doe_number', doe_number)

            for i in range(self.seeds_number):
                update_mooring_system(d_rope, d_chain , l_rope, z_fairlead, radius,
                            self.config_sim_file,  self.work_dir +os.sep +'Design_'+str(doe_number)+os.sep+'seed_no_'+str(i),
                            doe_number)
            
            



if __name__ == '__main__':

    # get path to current folder
    run_dir = os.path.dirname(os.path.realpath(__file__))

    of_model_dir = run_dir + os.sep + 'OpenFAST_Base_Case_Model'
    work_dir = run_dir + os.sep + 'Nautilus_15MW_Parametric_test'+os.sep
    config_env_design_file = run_dir + os.sep + 'Inputs_Definition'+os.sep+'DOE_Design_Environmental_Definitions_test.yaml'
    config_sim_file = run_dir + os.sep + 'Inputs_Definition'+os.sep+ 'Settings.yaml'
    


    wrapper = run_experiment(
        of_dir = of_model_dir,
        main_file="IEA-15-240-RWT-Nautilus.fst",
        work_dir=work_dir,
        config_env_design_file=config_env_design_file,
        config_sim_file=config_sim_file)

    mpi_options['mpi_run'] = False

    outputs = run_openfast(wrapper.fst_files,mpi_options)

    breakpoint()