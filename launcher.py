import os
from pathlib import Path

import io_utils as io
from input_creator import TerachemInput, TurbomoleInput
from sys_utils import JobPrepper


def launch_TCcalculation(folder: Path, geometry_file: Path, calc_settings: dict):
    '''This function preps and lanches a Terachem calculation given:
    candidate: the level of theory (e.g. 'casscf')
    folder: path to new folder that will contain the calculation
    geometry: path to xyz file
    general_settings: dictionary of settings that are common for a given molecule (e.g. charge, basis set..)
    candidate_settings: dictionary of settings specific to the calculation'''
    jobname = geometry_file.stem
    newjob = JobPrepper(folder, jobname=jobname)
    newjob.create_dir()
    newjob.copy_geometry_in_newdir(geometry_file)
    newjob.write_sbatch_short_TC()
    input = TerachemInput.from_default_TC(calc_settings)
    input.to_file(folder/'tc.in')
    with io.cd(folder):
        os.system('sbatch submit.sbatch')
        # os.system('export CUDA_VISIBLE_DEVICES=0,1; module load tc/23.08; nohup terachem tc.in > tc.out &')
        print(f'Launched {folder} on {geometry_file.stem}!')


def launch_TMcalculation(folder: Path, geometry_file: Path, n_singlets, charge):
    '''This function preps and lanches a EOM-CC2 TURBOMOLE calculation given:
    folder: path to the new folder that will contain the calculation
    geometry: path to xyz file
    general_settings: dictionary of settings that are common for a given molecule (e.g. charge, basis set..)
    candidate_settings: dictionary of settings specific to the calculation'''
    jobname = geometry_file.stem
    newjob = JobPrepper(folder, jobname=jobname)
    newjob.create_dir()
    newjob.copy_geometry_in_newdir(geometry_file)
    with io.cd(folder):
        tm_input = TurbomoleInput.from_default_eom(n_singlets, charge)
        tm_input.to_file('define-inputs.txt')
        newjob.prep_eom_calc()
        newjob.write_sbatch_TURBOMOLE()
        os.system('sbatch submit.sbatch')
        print(f'Launched {folder} on {geometry_file.stem}!')
