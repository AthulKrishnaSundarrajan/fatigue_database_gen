import os
import sys
import numpy as np

def check_mpi_env():
    """
    Determine if the environment variable governing MPI usage is set.

    Returns
    -------
    bool
        True if MPI is required, False if it's to be skipped, None if not set.
    """
    mpi_selection = os.environ.get('OPENMDAO_USE_MPI', None)

    # If OPENMDAO_USE_MPI is set to a postive value, the run will fail
    # immediately if the import fails
    if str(mpi_selection).lower() in ['always', '1', 'true', 'yes', 'y', 'on']:
        return True

    # If set to something else, no import is attempted.
    if mpi_selection is not None:
        return False

    # If unset, the import will be attempted but give no warning if it fails.
    return None

use_mpi = check_mpi_env()

if use_mpi is True:
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("Importing MPI failed and OPENMDAO_USE_MPI is true.")
elif use_mpi is False:
    MPI = None
else:
    try:
        from mpi4py import MPI

        # If the import succeeds, but it doesn't look like a parallel
        # run was intended, don't use MPI
        if MPI.COMM_WORLD.size == 1:
            MPI = None

    except ImportError:
        MPI = None

def map_comm_heirarchical(nFD, n_OF):
    """
    Heirarchical parallelization communicator mapping.  Assumes a number of top level processes
    equal to the number of finite differences, each
    with its associated number of openfast simulations.
    """
    N = nFD + nFD * n_OF
    comm_map_down = {}
    comm_map_up = {}
    color_map = [0] * nFD

    for i in range(nFD):
        comm_map_down[i] = [nFD + j + i * n_OF for j in range(n_OF)]
        color_map.extend([i + 1] * n_OF)

        for j in comm_map_down[i]:
            comm_map_up[j] = i

    return comm_map_down, comm_map_up, color_map

def subprocessor_loop(comm_map_up):
    """
    Subprocessors loop, waiting to receive a function and its arguements to evaluate.
    Output of the function is returned.  Loops until a stop signal is received

    Input data format:
    data[0] = function to be evaluated
    data[1] = [list of arguments]
    If the function to be evaluated does not fit this format, then a wrapper function
    should be created and passed, that handles the setup, argument assignment, etc
    for the actual function.

    Stop sigal:
    data[0] = False
    """
    # comm        = impl.world_comm()


    rank = MPI.COMM_WORLD.Get_rank()
    rank_target = comm_map_up[rank]

    keep_running = True
    while keep_running:
        data = MPI.COMM_WORLD.recv(source=(rank_target), tag=0)
        if data[0] == False:
            break
        else:
            func_execution = data[0]
            args = data[1]
            output = func_execution(args)
            MPI.COMM_WORLD.send(output, dest=(rank_target), tag=1)


def subprocessor_stop(comm_map_down):
    """
    Send stop signal to subprocessors
    """
    for rank in comm_map_down.keys():
        subranks = comm_map_down[rank]
        for subrank_i in subranks:
            MPI.COMM_WORLD.send([False], dest=subrank_i, tag=0)
        print("All MPI subranks closed.")