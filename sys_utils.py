import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobPrepper:
    new_dir: Path
    jobname: str

    def create_dir(self):
        if self.new_dir.is_dir():  # CANCELLAMI
            shutil.rmtree(self.new_dir)
        self.new_dir.mkdir()

    def copy_geometry_in_newdir(self, fn):
        shutil.copy2(fn, self.new_dir)

    def write_sbatch_short_TC(self):
        commands = f'''#!/usr/bin/env bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J cand
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

srun run.sh

'''
        with open(self.new_dir/'submit.sbatch', 'w') as sbatch_file:
            sbatch_file.write(commands)
        run_commands = '''#!/usr/bin/env bash

# Load necessary modules
module load tc/24.11

terachem tc.in > tc.out'''
        with open(os.open(self.new_dir/'run.sh', os.O_CREAT | os.O_WRONLY, 0o777), 'w') as run_file:
            run_file.write(run_commands)

    def write_sbatch_TURBOMOLE(self):
        commands = f'''#! /bin/bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J ref
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

module load turbomole/7.8
export PARNODES=2
export PARA_ARCH="SMP"
export TURBOMOLE_SYSNAME=x86_64-unknown-linux-gnu_smp

define < define-inputs.txt
dscf > dscf.out
ricc2 > ricc2.out


'''
        with open('submit.sbatch', 'w') as sbatch_file:
            sbatch_file.write(commands)

    def prep_eom_calc(self):
        os.system(f'module load turbomole/7.8; x2t {self.jobname}.xyz > coord; define < define-inputs.txt &> define.log')

import os
from pathlib import Path

def write_sbatch_grad_TC(directory: Path, job_name: str, gpu: bool = False, mem: str = "200G", time: str = "30:00:00"):
    sbatch_commands = f"""#!/usr/bin/env bash

#SBATCH -p elipierilab
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J {job_name}
#SBATCH --mem={mem}
#SBATCH -t {time}
{"#SBATCH --qos gpu_access" if gpu else ""}
{"#SBATCH --gres=gpu:1" if gpu else ""}

srun run.sh
"""
    sbatch_path = directory / 'submit.sbatch'
    with open(sbatch_path, 'w') as sbatch_file:
        sbatch_file.write(sbatch_commands)

    run_commands = '''#!/usr/bin/env bash

# Load necessary modules
module load tc/24.04

terachem tc.in > tc.out'''
    run_path = directory / 'run.sh'
    os.makedirs(directory, exist_ok=True)
    with open(os.open(run_path, os.O_CREAT | os.O_WRONLY, 0o777), 'w') as run_file:
        run_file.write(run_commands)

def main():
    d = {'new_dir': 'opt', 'jobname': 'benzene'}
    newjob = JobPrepper(**d)
    newjob.create_dir()
    newjob.copy_geometry_in_newdir('geom.xyz')
    newjob.write_sbatch_short_TC()


if __name__ == "__main__":
    main()
