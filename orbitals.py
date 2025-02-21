# Generate VMD scripts in each scr for CASSCF, CASCI, and hhtda methods.
import os
import re
import subprocess
import argparse
import io_utils as io
from pathlib import Path
from candidate import CandidateListGenerator

def get_geom_name(yaml_file):
    settings = io.yload(yaml_file)
    geometry = settings['general']['coordinates']
    geom_name = Path(geometry).stem
    return geom_name

def read_tc_in(tc_in_path):
    closed, active = None, None

    if not os.path.exists(tc_in_path):
        print(f"Warning: tc.in not found at {tc_in_path}")
        return None, None  # Return None if file doesn't exist

    with open(tc_in_path, "r") as f:
        for line in f:
            if match := re.search(r"closed\s+(\d+)", line):
                closed = int(match.group(1))
            if match := re.search(r"active\s+(\d+)", line):
                active = int(match.group(1))

    if closed is None or active is None:
        print(f"Warning: Could not extract orbitals from {tc_in_path}")

    return closed, active

def parse_molden(molden_path):
    if not os.path.exists(molden_path):
        print(f"File not found: {molden_path}")
        return []

    orbitals = []
    in_mo_section = False

    print(f"Reading {molden_path}")

    # Read first few lines for debugging
    with open(molden_path, "r") as f:
        lines = f.readlines()
    
    print("\n".join(lines[:10]))

    for line in lines:
        if "[MO]" in line:
            in_mo_section = True
            continue
        if in_mo_section:
            if "Ene=" in line:  # Use "Ene=" as the orbital marker
                orbitals.append(len(orbitals) + 1)  # Count orbitals

    print(f"Found {len(orbitals)} orbitals in Molden file")
    return orbitals

def orbital_labels(dir_name, orbitals, closed, active):
    #First, check if hhTDA or CASSCF/CASCI
    is_hhtda = "hhtda" in dir_name.lower()

    #HOMO/LUMO labels for all printed orbitals
    match =re.search(r'AS(\d{1,2})(\d{1,2})$', dir_name)
    if not match and not is_hhtda:
        print("Could NOT find AS in directory, make sure it is CASSCF/CASCI method")
        return f"NaN"

    if not is_hhtda:
        num_e = int(match.group(1))
        print(f"num_e = {num_e}")
        lowest_E_active = closed + 1 #Lowest energy active orbital
        electrons_fill = num_e // 2

    if is_hhtda:
        HOMO = active - 1
        LUMO = HOMO + 1
    else:
        HOMO = lowest_E_active + electrons_fill  -1
        LUMO = HOMO + 1

    if orbitals == HOMO:
        label = "HOMO"
    elif orbitals == LUMO:
        label = "LUMO"
    elif orbitals < HOMO:
        label = f"HOMO-{HOMO - orbitals}"
    elif orbitals > LUMO:
        label = f"LUMO+{orbitals - LUMO}"
    else:
        label = f"NaN" 

    return label

def generate_vmd_script(vmd_file, scr_dir, molden_file, orbital_range, dir_name, closed, active):    
    #Generates a VMD script for visualizing orbitals
    orb_folder = os.path.join(scr_dir, "orbitals")
    os.makedirs(orb_folder, exist_ok=True)

    with open(vmd_file, 'w') as vmd_script:
        vmd_script.write("#!/usr/local/bin/vmd \n")
        vmd_script.write("display depthcue off \n")
        vmd_script.write("color Display Background white \n")
        vmd_script.write("axes location off \n")
        vmd_script.write("proc vmdrestoremycolors {} { \n")
        vmd_script.write("color scale colors RWB {1.0 0.0 0.0} {1.0 1.0 1.0} {0.0 0.0 1.0}\n} \n")
        vmd_script.write("}\n")

        vmd_script.write(f"mol new {molden_file} type molden \n")
        
        vmd_script.write("mol addrep 0  \n")
        vmd_script.write("mol modstyle 0 0 DynamicBonds 1.600000 0.100000 12.000000 \n")
        vmd_script.write("mol modcolor 0 0 Element  \n")

        for orbital in orbital_range:
            label = orbital_labels(dir_name, orbital, closed, active)
            vmd_script.write(f"mol changeframe {orbital}\n")
            
            # **First isosurface (Positive 0.05, Color: Red)**
            vmd_script.write("mol selection all \n")
            vmd_script.write("mol material Translucent \n")
            vmd_script.write("mol addrep 0 \n")
            vmd_script.write("mol modcolor 1 0 ColorID 1  \n")
            vmd_script.write(f"mol modstyle 1 0 Orbital 0.050000 {orbital} 0 0 0.075 1 6 0 0 1  \n")
            
            # **Second isosurface (Negative -0.05, Color: Blue)**
            vmd_script.write("mol selection all \n")
            vmd_script.write("mol material Translucent \n")
            vmd_script.write("mol addrep 0 \n")
            vmd_script.write(f"mol modstyle 2 0 Orbital -0.050000 {orbital} 0 0 0.075 1 6 0 0 1  \n")
            vmd_script.write("mol modcolor 2 0 ColorID 0  \n")

            vmd_script.write(f'render TachyonInternal "{label}_Orbital{orbital}.tga" \n')

        vmd_script.write("exit \n")

    print(f"Generated VMD script: {vmd_file}")
    return vmd_file

def vmd_initial(scr_dir, vmd_file):
    """Runs the VMD script to generate orbital images."""
    command = ["vmd", "-e", vmd_file]
    print(f"Running VMD script: {' '.join(command)}")
    orb_dir = os.path.join(scr_dir, "orbitals")
    os.makedirs(orb_dir, exist_ok = True)
    subprocess.run(command, cwd=orb_dir, check=True)

def process_cand(yaml_file):
    settings = io.yload(yaml_file)
    geom = get_geom_name(yaml_file)
    base = os.getcwd()

    for method, method_settings in settings['candidates'].items():
        print(f"Processing method: {method}")

        for cand_folder in os.listdir(base):
            cand_path = os.path.join(base, cand_folder)

            if not os.path.exists(cand_path) or not cand_folder.startswith(method):
                continue

            scr_dir = os.path.join(cand_path, f"scr.{geom}")
            molden_path = os.path.join(scr_dir, f"{geom}.molden")
            tc_in_path = os.path.join(cand_path, "tc.in")

            if not os.path.exists(tc_in_path):
                print(f"Skipping {cand_path}, tc.in not found.")
                continue

            closed, active = read_tc_in(tc_in_path)
            if closed is None or active is None:
                print(f"Skipping {cand_path}, could not extract orbitals.")
                continue

            if "hhtda" in method.lower():
                orbital_range = list(range(max(1, active - 10), active + 11))
            else:
                orbital_range = list(range(closed + 1, closed + 1 + active))

            print(f"Candidate: {cand_folder}, Closed: {closed}, Active: {active}, Orbitals: {orbital_range}")
            
            if os.path.exists(scr_dir):
                print(f"Processing {molden_path} with orbitals {orbital_range}")

                # Parse the Molden file
                available_orbitals = parse_molden(molden_path)
                if not available_orbitals:
                    print(f"Skipping {cand_path}, no orbitals found in Molden file.")
                    continue

                # Ensure only valid orbitals are selected
                orbital_range = [orb for orb in orbital_range if orb in available_orbitals]
                if not orbital_range:
                    print(f"No valid orbitals found for {cand_path}")
                    continue
            
                vmd_file = os.path.join(scr_dir, "orbitals", f"{cand_folder}.vmd")
                os.makedirs(os.path.dirname(vmd_file), exist_ok = True)
                generate_vmd_script(vmd_file, scr_dir, molden_path, orbital_range, cand_folder, closed, active) 
                vmd_initial(scr_dir, vmd_file)
            else:
                print(f"scr directory not found in {cand_folder}")

def main():
    parser = argparse.ArgumentParser(description="Generate VMD scripts for orbital visualization.")
    parser.add_argument("-i", "--input_yaml", type=str, required=True, help="Path to the input YAML file.")
    args = parser.parse_args()
    yaml_file = args.input_yaml

    process_cand(yaml_file)

if __name__ == "__main__":
    main()
