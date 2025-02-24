import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml


@contextmanager
def cd(newdir):
    '''
    This is used to enter in a folder with 'with cd(folder):' and send a command, then leave the folder.
    '''
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def yload(fn):
    """This function takes a yaml filename as input and returns the settings"""
    with open(fn, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)
    return settings


def frames_counter(fn):
    '''
    Given a trajectory files, gives back number of frame and number of atoms.
    fn :: FilePath
    '''
    f = open(fn)
    atomN = int(f.readline())
    f.close()
    with open(fn) as f:
        for i, l in enumerate(f):
            pass
    frameN = int((i + 1)/(atomN+2))
    return (atomN, frameN)


def read_trajectory(fn):
    '''
    reads a md.xyz format file and gives back a dictionary with
    geometries and all the rest
    '''
    atomsN, frameN = frames_counter(fn)
    geom = np.empty((frameN, atomsN, 3))
    atomT = []
    with open(fn) as f:
        for i in range(frameN):
            f.readline()
            f.readline()
            for j in range(atomsN):
                a = f.readline()
                bb = a.split(" ")
                b = [x for x in bb if x != '']
                geom[i, j] = [float(b[1]), float(b[3]), float(b[2])]
                if i == 0:
                    atomT.append(b[0])
    final_data = {
                 'geoms': geom,
                 'atomsN': atomsN,
                 'frameN': frameN,
                 'atomT': atomT,
                 }
    return final_data


def save_traj(arrayTraj, labels, fn):
    '''
    given a numpy array of multiple coordinates, it prints the concatenated xyz file
    arrayTraj :: np.array(ncoord,natom,3)    <- the coordinates
    labels :: [String] <- ['C', 'H', 'Cl']
    filename :: String <- filepath
    '''
    (ncoord, natom, _) = arrayTraj.shape
    string = ''
    for geo in range(ncoord):
        string += str(natom) + '\n\n'
        for i in range(natom):
            string += "   ".join([labels[i]] + ['{:10.10f}'.format(num) for num in arrayTraj[geo, i]]) + '\n'

    with open(fn, "w") as myfile:
        myfile.write(string)


def write_last_frame_to_file(traj: dict, name: Path):
    geoms = np.array([traj['geoms'][-1]])
    save_traj(geoms, traj['atomT'], name)


def dynamic_check_opt_status(fo, fc, stability_wait_time=60, max_wait_time=100000):
    """
    Checks if the optimization is complete based on file size stability and convergence message.
    
    Args:
        fo (str): Path to the primary output file.
        fc (str): Path to the checkpoint or final structure file.
        stability_wait_time (int): Time in seconds for file size to remain stable.
        max_wait_time (int): Maximum wait time in seconds before assuming completion.
        
    Returns:
        bool: True if optimization appears complete based on file size stability and convergence message; False otherwise.
    """
    
    # Wait until both files exist
    while not (os.path.exists(fo) and os.path.exists(fc)):
        time.sleep(5)
    
    print("Optimization files detected. Monitoring for completion...")

    start_time = time.time()
    stable_start_time = None

    while True:
        file1_size = os.stat(fo).st_size
        time.sleep(30)
        file2_size = os.stat(fo).st_size

        # Check if file size is stable
        if file1_size == file2_size:
            # Start timing the stability period
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= stability_wait_time:
                # Check for the convergence message after file stability
                converged = grep_string_in_file("Converged!", fo)
                if converged:
                    print("The optimization converged successfully!")
                else:
                    print("The optimization had problems or did not converge. Check the output.")
                return converged
        else:
            # Reset stability timer if file size changes
            stable_start_time = None
        
        # Check for timeout to prevent indefinite waiting
        if time.time() - start_time > max_wait_time:
            print("Warning: File size check timed out. Proceeding as if optimization is complete.")
            return False

        time.sleep(1)

def grep_string_in_file(search_string, file_path):
    """
    Searches for a specific string in a file.
    
    Args:
        search_string (str): The string to search for in the file.
        file_path (str): Path to the file to search.
        
    Returns:
        bool: True if the string is found; False otherwise.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if search_string in line:
                    return True
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return False


def check_calc_status(fn):
    status = grep_string_in_file("Job finished:", fn)
    if not status:
        print(f'Check {fn}, there is some problem!')
    return status
