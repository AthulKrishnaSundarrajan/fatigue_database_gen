# MoorPy Example Script:
# Example of manually setting up a mooring system in MoorPy and solving equilibrium.

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

def update_mooring_system (polyester_nom_diameter, chain_nom_diameter, l_rope, z_fairlead, radius, file_settings, save_dir, doe_number): 
    
    import numpy as np
    import matplotlib.pyplot as plt
    import moorpy as mp
    from moorpy.MoorProps import getLineProps
    import os
    import math


    # ----- choose some system geometry parameters -----
    fixed_variables = _load_yaml(file_settings)

    depth     = fixed_variables["mooring_layout"]["depth"]  
    angles    = np.radians(fixed_variables["mooring_layout"]["angles"])      # line headings list [rad]
    zFair     = z_fairlead                                                   # fairlead z elevation [m]
    lineLength_bottom_chain = 0.273 * (0.9914*radius)
    lineLength_top_chain = (0.9914*radius) - lineLength_bottom_chain - l_rope                          
    typeName_chain  = str(fixed_variables["mooring_layout"]["typeName_chain"])                        # identifier string for the line type
    typeName_rope  = str(fixed_variables["mooring_layout"]["typeName_rope"])                         # identifier string for the line type

    #nb_segments_top_chain = fixed_variables["mooring_layout"]["nb_segments_top_chain"]
    nb_segments_top_chain = math.ceil(lineLength_top_chain/5)
    #nb_segments_rope = fixed_variables["mooring_layout"]["nb_segments_rope"]
    nb_segments_rope = math.ceil(l_rope/15)
    nb_segments_bottom_chain = fixed_variables["mooring_layout"]["nb_segments_bottom_chain"]

    x_fairlead = fixed_variables["mooring_layout"]["x_fairlead_line_1"] #m
    y_fairlead = fixed_variables["mooring_layout"]["y_fairlead_line_1"] #m 

    x_anchor = -radius * np.cos(np.radians(45)) #m
    y_anchor = radius * np.sin(np.radians(45)) #m 
    z_anchor = - depth #m

    z_buoy = fixed_variables["mooring_layout"]["z_in_line_buoy_wr_seafloor"]


    # ----- set up the mooring system and floating body -----

    # Create new MoorPy System and set its depth
    ms = mp.System(depth=depth)

    ms.activateDynamicStiffness = False
    ms.MDoptions = dict(dtM=fixed_variables["mooring_time_solver"]["dtM"], kb=fixed_variables["mooring_time_solver"]["kb"], cb = fixed_variables["mooring_time_solver"]["cb"], 
                        dtIC = fixed_variables["mooring_time_solver"]["dtIC"], TmaxIC=fixed_variables["mooring_time_solver"]["TmaxIC"], threshIC = fixed_variables["mooring_time_solver"]["threshIC"], 
                        dtOut = fixed_variables["mooring_time_solver"]["dtOut"], CdScaleIC = fixed_variables["mooring_time_solver"]["CdScaleIC"]) 

    # add a line type
    ms.setLineType(dnommm=150, material=typeName_chain, name=typeName_chain)  # this would be 120 mm chain
    ms.setLineType(dnommm=226, material=typeName_rope, name=typeName_rope)  # this would be 120 mm chain

    # For each line heading, set the anchor point, the fairlead point, and the line itself
    for i, angle in enumerate(angles):
        
        inclinaison_angle = np.asin((depth - z_buoy - np.abs(zFair))/(lineLength_top_chain + l_rope))

        x_1 = x_fairlead - lineLength_top_chain*np.cos(inclinaison_angle)*np.sqrt(2)/2
        y_1 = y_fairlead + lineLength_top_chain*np.cos(inclinaison_angle)*np.sqrt(2)/2
        z_1 = z_fairlead - np.sqrt(lineLength_top_chain**2 - (x_1-x_fairlead)**2 - (y_1 - y_fairlead)**2)
        
        x_2 = x_fairlead - (lineLength_top_chain + l_rope)*np.cos(inclinaison_angle)*np.sqrt(2)/2
        y_2 = y_fairlead + (lineLength_top_chain + l_rope)*np.cos(inclinaison_angle)*np.sqrt(2)/2
        z_2 = z_fairlead - np.sqrt((lineLength_top_chain + l_rope)**2 - (x_2-x_fairlead)**2 - (y_2 - y_fairlead)**2)

        # create end Points for the line
        ms.addPoint(-1, [x_fairlead*np.sign(np.cos(angle)), y_fairlead*np.sign(np.sin(angle)), zFair])   # create fairlead point (type 0, fixed)
        ms.addPoint(0, [x_1*np.sign(np.cos(angle)), y_1*np.sign(np.sin(angle)), z_1]) 
        ms.addPoint(0, [x_2*np.sign(np.cos(angle)), y_2*np.sign(np.sin(angle)), z_2])
        ms.addPoint(1, [x_anchor*np.sign(np.cos(angle)), y_anchor*np.sign(np.sin(angle)), z_anchor])   # create fairlead point (type 0, fixed)

        # add a Line going between the anchor and fairlead Points
        ms.addLine(lineLength_top_chain, typeName_chain, nSegs = nb_segments_top_chain, pointA=4*i+2, pointB=4*i+1)
        ms.addLine(l_rope, typeName_rope, nSegs = nb_segments_rope,  pointA=4*i+3, pointB=4*i+2)
        ms.addLine(lineLength_bottom_chain, typeName_chain, nSegs = nb_segments_bottom_chain,  pointA=4*i+4, pointB=4*i+3)
        
    # ----- Add the in-line buoy

    for i in [2, 6, 10, 14]:
        ms.pointList[i].CdA = fixed_variables["mooring_layout"]["in_line_buoy_CdA"]
        ms.pointList[i].Ca = fixed_variables["mooring_layout"]["in_line_buoy_Ca"]
        ms.pointList[i].v = fixed_variables["mooring_layout"]["in_line_buoy_volume"]
        ms.pointList[i].m = fixed_variables["mooring_layout"]["in_line_buoy_mass"]

    # ----- run the model to demonstrate -----
    file_location = os.getcwd() + '/'
    file_name = "IEA-15-240-RWT-Nautilus_MoorDyn_synthetic" + "_design_" + str(doe_number) + ".dat"
    
    #ms.initialize()                                             # make sure everything's connected

    #ms.solveEquilibrium()                                       # equilibrate

    ms.unload(file_location + file_name, MDversion=2, flag = 'ts', 
              outputList = ['LINE2PZ','LINE5PZ', 'LINE8PZ', 'LINE11PZ'], Lm = 10)                         # export to MD input file


    #%% Opening the file and change some lines

    critical_damping_ratio = 1.0

    chain_name = typeName_chain 
    chain_vol_diameter = (1.8 * chain_nom_diameter)/1000
    chain_mass_dens = 19.9 * 1E+03 * (chain_nom_diameter/1000)**2
    print('chain_density', chain_mass_dens)
    chain_EA = 85.4 * 1E+09 * (chain_nom_diameter/1000)**2
    chain_BA = critical_damping_ratio * lineLength_top_chain/nb_segments_top_chain * np.sqrt(chain_EA * chain_mass_dens)
    chain_EI = 0
    chain_Cd = fixed_variables["mooring_layout"]["chain_Cd"]
    chain_Ca = fixed_variables["mooring_layout"]["chain_Ca"]
    chain_CdAx = 0.0
    chain_CaAx = 0.0

    polyester_name = typeName_rope 
    polyester_vol_diameter = 0.86 * polyester_nom_diameter/1000
    polyester_mass_dens = 0.0005484 * (polyester_nom_diameter)**2 + 0.0317656 * (polyester_nom_diameter) + 5.6331598
    polyester_MBL = (0.293466 * (polyester_nom_diameter)**2 + 0.206689 * (polyester_nom_diameter) + 342.624548) * 1000
    polyester_EAqs = 14.8 * polyester_MBL
    polyester_EAd = 17.6619 * polyester_MBL
    polyester_EAd_Lm = 24.3671
    polyester_BAqs = critical_damping_ratio * l_rope/nb_segments_rope * np.sqrt(polyester_EAqs * polyester_mass_dens)
    polyester_BAd = critical_damping_ratio * l_rope/nb_segments_rope * np.sqrt((polyester_EAd + fixed_variables["mooring_layout"]["mean_load"]*polyester_EAd_Lm) * polyester_mass_dens)
    polyester_EI = 0
    polyester_Cd = fixed_variables["mooring_layout"]["polyester_Cd"]
    polyester_Ca = fixed_variables["mooring_layout"]["polyester_Ca"]
    polyester_CdAx = 0.0
    polyester_CaAx = 0.0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Creates the folder (and parent directories if needed)
        
    final_file_name = "IEA-15-240-RWT-Nautilus_MoorDyn_synthetic" + "_design_" + str(doe_number) + ".dat"
    f_in = open(file_location + file_name, 'rt')
    f_out = open(save_dir + final_file_name, 'w')

    for k, line in enumerate(f_in):
        if k == 5:
            f_out.write(str(chain_name) + '  ' + str(chain_vol_diameter) + '  ' + str(chain_mass_dens) 
                    + '  ' + str(chain_EA)  + '  ' + str(chain_BA) + ' ' + str(chain_EI) + '  ' 
                    + str(chain_Cd) + '  ' + str(chain_Ca) + '  ' + str(chain_CdAx) + '  ' 
                    + str(chain_CaAx) + '\n')
        elif k == 6:
            f_out.write(str(polyester_name) + '  ' + str(polyester_vol_diameter) + '  ' + str(polyester_mass_dens) 
                + '  ' + str(polyester_EAqs) + '|' + str(polyester_EAd) + '|' + str(polyester_EAd_Lm) + '  ' 
                + str(polyester_BAqs) + '|' + str(polyester_BAd) + ' ' + str(polyester_EI) + '  ' 
                + str(polyester_Cd) + '  ' + str(polyester_Ca) + '  ' + str(polyester_CdAx) + '  ' 
                + str(polyester_CaAx) + '\n')
        else:
            f_out.write(line)
    f_in.close()
    f_out.close()
    
    if os.path.exists(file_location + file_name):
        os.remove(file_location + file_name)  # Deletes the file
        print("File deleted successfully")
    else:
        print("File does not exist")

if __name__ == "__main__":
    update_mooring_system(polyester_nom_diameter = 226, chain_nom_diameter = 150 , l_rope = 764.64, 
                          file_settings = "../Inputs_Definition/Settings.yaml",
                          save_dir = "../Inputs_Definition/Mooring_Designs_Input_Files/",
                          doe_number=1)