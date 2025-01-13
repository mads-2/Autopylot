'''
Task of this script
1. Go into the scratch folder of every candidate
2. Determine the # of orbitals for the active space. If no active space, pick HOMO-2 to LUMO+2 (orbkit skips the highest orbital so +1
3. Run orbkit command (example: orbkit -i h2o.molden -o vis/h2o --otype=vmd --adjust_grid=5 0.5 --calc_mo=homo-1:lumo+2)
4. Go back into every scratch folder, and run the vmd command (oh yea, this requires VMD) vmd -e vis/h2o_MO.vmd
'''

import os
import subprocess

def find_orbital_range:(scratch_dir, AS=None):

    if AS:
        return AS

    # Default range for hhTDA methods
    low = 'homo-2'
    high = 'lumo+3'
    return f"{low}:{high}"

def orbkit_initial(scratch_dir, molden_file, orbital_range):
    
    vis_folder = os.path.join(scratch_dir, "vis"
    os.makdir(vis_folder, exist_ok=True) 
    output = os.path.join(vis_folder) 

    command = [ 
        "orbkit",
        "-i", molden_file,
        "-o", output,
        "--otype=vmd",
        "--adjust_grid=5", "0.05",
        f"--calc_mo{orbital_range}"
    ]
    print(f"Running orbkit: {' '.join(command)}")

def vmd_initial(scratch_dir, vmd_file):
    
    command = ["vmd", "-e", vmdfile]
    print (f"Running VMD script, Generating pictures: {' '.join(command)}")
    subproces.run(command, cwd=scratchdir/vis, check=True)

def process_cand(base_folder, AS=None):
    
    for cand_dir in on.listdir(base_dir):
        cand_path = os.join.path(base_dir, cand_dir)
        if not os.path.isdir(cand_path):
            continue

        scratch_dir = os.path.join(cand_path, scr.*)
        if not os.path.exists(scratch_dir):
            continue

        #Searches all candidates for {cand_name}.molden files
        molden_group = [f for f in os.lisdir(scratch_dir_ if f.endswith(".molden")]
        if not molden_file:
            print(f"No molden file found in {scratch_dir}")

        #finds the singular {cand_name}.molden file per cand
        molden_single = molden_files[0]
        print(f" Found Molden file: {molden_file} in {scratch_dir}")

        #Determine orbitals needed from AS
        orbital_range = find_orbital_range(scratch_dir, AS)

        try:
            orbitkit_initial(scratch_dir, molden_file, orbital_range)
        except subprocess.CalledProcessError as e: 
            print (f"Error running orbkit in {scratch_folder}: {e}")

        vis_folder = os.path




        


