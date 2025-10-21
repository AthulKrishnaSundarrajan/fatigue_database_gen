import os
import numpy as np
import shutil
from wind_generation import mann_model

from FAST_wrapper import FAST_wrapper
from pCrunch import Crunch,FatigueParams, AeroelasticOutput,read

of_path = os.path.realpath(shutil.which('openfast') )

fail_value = 9999
allow_fails = True
fatigue_channels = {'TwrBsMyt': FatigueParams(slope=4),}

def run_wrapper(fst_file,wind_dir,wind_dict,overwrite_flag):

    print('Generating wind files')

    mann_model(wind_dict['U_hub'],
     wind_dict['nx'], wind_dict['ny'], wind_dict['nz'],
     wind_dict['dx'], wind_dict['dy'], wind_dict['dz'],
     wind_dict['sim_time'],wind_dict['file_name'], wind_dict['wind_seed'])

    wrapper = FAST_wrapper()

    wrapper.FAST_exe = of_path

    FAST_directory,FAST_InputFile = os.path.split(fst_file)

    wrapper.FAST_directory = FAST_directory
    wrapper.FAST_InputFile = FAST_InputFile

    FAST_Output = os.path.join(FAST_directory, FAST_InputFile[:-3]+'outb')

    output_flag = (os.path.exists(FAST_Output))

    if overwrite_flag or (not overwrite_flag and not output_flag):
        failed = wrapper.execute()
        if failed:
            print('OpenFAST Failed! Please check the run logs.')
            if allow_fails:
                print(f'OpenFAST failures are allowed. All outputs set to {fail_value}')
            else:
                raise Exception('OpenFAST Failed! Please check the run logs.')

    else:

        failed = False
        print('OpenFAST not executed: Output file "%s" already exists. To overwrite this output file, set "overwrite_outfiles = True".'%FAST_Output)

    print('removing wind directory '+wind_dir)
    shutil.rmtree(wind_dir)

    output = FAST_Output

    return output


def run_serial(fst_files,wind_dirs,wind_list,overwrite_flag):

    output_list = []
    # run openfast simulations in parallel
    for i_file,fst_file in enumerate(fst_files):
        output = run_wrapper(fst_file,wind_dirs[i_file],wind_list[i_file],overwrite_flag)

        output_list.append(output)

    return output_list

def evaluate_multi(case_data):
    print(case_data['fst_file'])

    output = run_wrapper(case_data['fst_file'],case_data['wind_dir'],case_data['wind_dict'],case_data['overwrite_flag'])

    return output



def run_mpi(case_data_all,mpi_options):

    from openmdao.utils.mpi import MPI

    # mpi comm management
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sub_ranks = mpi_options['mpi_comm_map_down'][rank]

    size = len(sub_ranks)

    N_cases = len(case_data_all)
    N_loops = int(np.ceil(float(N_cases)/float(size)))

    output_list = []

    for i in range(N_loops):
        idx_s    = i*size
        idx_e    = min((i+1)*size, N_cases)

        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            data   = [evaluate_multi, case_data]
            rank_j = sub_ranks[j]
            comm.send(data, dest=rank_j, tag=0)

        # for rank_j in sub_ranks:
        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            rank_j = sub_ranks[j]
            data_out = comm.recv(source=rank_j, tag=1)
            output_list.append(data_out)

    return output_list


def run_openfast(fst_files,mpi_options,wind_dirs,wind_list,overwrite_flag = True):

    if mpi_options['mpi_run']:
        # evaluate the closed loop simulations in parallel using MPI

        case_data_all = []
        for i_file,fst_file in enumerate(fst_files):
            case_data = {}
            case_data['fst_file'] = fst_file
            case_data['overwrite_flag'] = overwrite_flag
            case_data['wind_dir'] = wind_dirs[i_file]
            case_data['wind_dict'] = wind_list[i_file]

            case_data_all.append(case_data)

        sim_outputs = run_mpi(case_data_all,mpi_options)

    else:

        # evaluate the closed loop simulations serially
        sim_outputs = run_serial(fst_files,wind_dirs,wind_list,overwrite_flag)


    return sim_outputs