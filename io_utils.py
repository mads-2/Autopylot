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


def dynamic_check_opt_status(fo, fc):
    while not (os.path.exists(fo) and os.path.exists(fc)):
        time.sleep(1)
    finished = False
    while not finished:
        file1 = os.stat(fo)
        file1_size = file1.st_size
        time.sleep(1)
        file2 = os.stat(fo)
        file2_size = file2.st_size
        comp = file2_size - file1_size
        if comp == 0:
            converged = grep_string_in_file("Converged!", fo)
            finished = True
        else:
            time.sleep(10)
    if converged:
        print('The optimization converged!')
    else:
        print('The optimization had problems. Check it!')
    return converged


def grep_string_in_file(pattern, fn):
    '''This function checks if a string is present in a file.
    It takes a string and a filename as input
    and returns a boolean (True=string found in file)'''
    textfile = open(fn, 'r')
    filetext = textfile.read()
    string_found_in_file = re.findall(pattern, filetext)
    return string_found_in_file


def check_calc_status(fn):
    status = grep_string_in_file("Job finished:", fn)
    if not status:
        print(f'Check {fn}, there is some problem!')
    return status
