# Autopylot: Pragmatic Benchmarking of Excited-State Electronic Structure

```text
   +--------------+
   |.------------.|
   ||   \ | /    ||
   ||   ~ O ~    ||
   ||   / | \    ||
   ||            ||
   |+------------+|
   +-..--------..-+
   .--------------.
  / /============\ \
 / /==============\ \
/____________________\
\____________________/
```

Paper: https://pubmed.ncbi.nlm.nih.gov/40652391/  
Lab:   https://www.pierilab.com  
Repo:  https://github.com/mads-2/Autopylot  

---

## My Contributions

As **second author** on this project, I contributed significantly to both the computational workflow and the publication.  
My role emphasized **workflow automation, large-scale data handling, dimensionality reduction, and visualization** — skills directly applicable to data science.  

- **Large-scale workflow execution:** Ran Automated benchmarking pipeline across **28 small organic molecules** involving geometry optimization, single-point energy calculations, numerical gradients, and the grader.  
- **Data management at scale:** Organized and standardized thousands of output files into structured, machine-readable formats for downstream analysis. Debugged and stabilized code to handle **edge cases in messy scientific data**.  
- **Dimensionality reduction & model comparison:** Extended Autopylot beyond simple method/basis/active space sweeps to include **temperature (FON) and range-separation parameter (ω)** 
- **High-performance computing (HPC):** Deployed workflows on UNC’s **Longleaf cluster**, managing resource allocation, batch submission scripts, and **parallel job orchestration** across GPUs.  
- **Reproducibility & environments:** Ensured **Conda + YAML-based reproducibility**, enabling others to recreate the entire workflow with minimal setup.  
- **Visualization & communication:**  
  - Designed **publication-quality figures** and **workflow schematics** using Inkscape.  
  - Created the **visual abstract** for the paper.  
  - Generated automated plots (method rankings, spectra overlays) that reduced manual interpretation effort.  
- **Supplementary Information curation:** Managed and compressed **58+ GB of orbital data** using Git LFS; documented procedures for efficient access and recombination.  
- **External API integration:** Troubleshot and stabilized workflows that relied on **TeraChem Protocol Buffers (TCPB)** — a client–server API for high-performance quantum chemistry — ensuring smooth integration into the larger pipeline.  

### Skills Built
Python • Data Visualization • Workflow Automation • Dimensionality Reduction • Reproducibility • HPC (Slurm, GPUs) • API Integration (TCPB) • Large-Scale Data Management • Scientific Communication  

---

# Autopylot Workflow Description

### Definition of work, and step-by-step instructions 
First, we need to create environment. Type the command: `conda env create -f environment.yml`.\
After that, before running the package, we need to activate Autopylot environment on cluster: `ml anaconda`, following by `conda activate autopylot`.\
There are a lot of python codes in the package we are able to use. The code `launcher.py` contains the functions calling GPU or CPU computations. Currently, only Terachem and Turbomole computations are available. Additional launchers may be added in the future.\
The program `candidate.py` contains functions that write Terachem input file. For now, it contains CASSCF, FOMO-CASCI, and hhTDA\
The `test_input.yaml` file contains information of active space (CASSCF, FOMO-CASCI), FON temperature (FOMO-CASCI and hhTDA), and range correction (hhTDA).\
The program `make_sh.py` allows you to set the submission scripts so that it can be run on any machine. This program directly changes the `sys_utils.py` which controls the duplicating of submission scripts and execution of jobs.\
The program `autopilot.py` launches the Terachem and Turbomole computations. When running this, please type `python autopilot.py -i test_input.yaml`. This will automatically generate and launch a bunch of computations on GPU nodes, while writing one folder for each.\
The program `gradient.py` will run gradient calcuations from the candidates that ran through `autopilot.py` sucessfully, gradient time is used in the grader as it is a better indicator of candidate cost than the energies of n states.\
The program `spectra_grader.py` plots and grades each results automatically. Run with `python spectra_grader.py -i test_input.yaml`.

### Step-by-Step workflow
I recommend running calculations in a directory seperate from your Autopylot directory to keep things clean and organinzed. As a result I will mention /path/to/Autopylot/directory/ as a stand-in for your compiled autopylot path.\
0. Create the envirnoment if you haven't done so already: `conda env create -f environment.yml`\
1. Activate Autopylot: In terminal, type: `ml anaconda`, then `conda activate autopylot`. You may elect to optimize S0 inside of autopilot.py first. To do this, please see `test_input_with_opt.yaml` for setup\
2. Set your submission scripts to work with your machine: `python /path/to/Autopylot/directory/make_sh.py candidate_sh.txt ref_sh.txt`. Example `candidate_sh.txt` and `ref_sh.txt` are provided. You only need to do this once every time you need to use differnt settings on your machine\ 
3. Run the autopilot program: `python /path/to/Autopylot/directory/autopilot.py -i test_input.yaml`\
4. Once autopilot calulations are done: Run gradient.py: `python /path/to/Autopylot/directory/gradient.py -i test_input.yaml`\
5. Once gradient calulations are complete and `gradient.py` is done: Run the grader and collect results: `python /path/to/Autopylot/directory/spectra_grader.py -i test_input.yaml`\

## Helpful Tips
If you would like to still have access to your terminal command line while gradient.py is running, put the command into a submission script in your machine when available. I personally do this for each step as a safe practice. 

## Visualizing your results 
Organized scores will be in `Final_Scores.txt`.\
The plot with state energies and grader scores can be found in `{title}_results.png`.\
`spectra_grader.py` will generate a directory `{title}_UVVis_plots` directory: here the plot represent the aligments of each state per candidate against the reference of same said candidate.

