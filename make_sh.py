import argparse
from pathlib import Path

def replace_commands_block(template_path, sys_utils_py="sys_utils.py", block_index=1):
    # Determine the directory of this script
    script_dir = Path(__file__).parent
    sys_utils_path = script_dir / sys_utils_py

    # Read the template content
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()

    # Read the content of sys_utils.py
    with open(sys_utils_path, 'r') as sys_utils_file:
        code_content = sys_utils_file.read()

    # Locate all `commands = f'''` blocks
    split_blocks = code_content.split("commands = f'''")
    
    # Check if the specified block index exists
    if block_index >= len(split_blocks):
        print(f"Error: Block index {block_index} not found in {sys_utils_py}.")
        return

    # Replace the specified block content with the new template
    split_blocks[block_index] = f"{template_content}\n'''" + split_blocks[block_index].split("'''", 1)[1]

    # Reconstruct the updated content
    updated_content = "commands = f'''".join(split_blocks)

    # Write the updated content back to sys_utils.py
    with open(sys_utils_path, 'w') as sys_utils_file:
        sys_utils_file.write(updated_content)

    print(f"Updated block index {block_index} in sys_utils.py.")

if __name__ == "__main__":
    # Set up argument parsing for the template files
    parser = argparse.ArgumentParser(description="Modifies specific commands blocks in sys_utils.py.")
    parser.add_argument("cand_template", help="Path to the candidate sbatch template file")
    parser.add_argument("ref_template", help="Path to the reference sbatch template file")
    args = parser.parse_args()

    # Paths for the candidate and reference templates
    cand_template_path = Path(args.cand_template)
    ref_template_path = Path(args.ref_template)

    # Check if the template files exist
    if not cand_template_path.is_file():
        print(f"Error: Candidate template file '{cand_template_path}' not found.")
    elif not ref_template_path.is_file():
        print(f"Error: Reference template file '{ref_template_path}' not found.")
    else:
        # Enforce order: apply candidate template changes first (block index 1), then reference template changes (block index 2)
        replace_commands_block(cand_template_path, block_index=1)
        replace_commands_block(ref_template_path, block_index=2)

