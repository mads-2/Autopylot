# Autopilot Workflow Description
### Defininton of work, and step-by-step instructions 
First, we need to create environment. Type the command: `conda env create -f environment.yml`.\
After that, before running the package, we need to activate autopilot environment on cluster: `ml anaconda`, following by `conda activate autopilot`.\
There are a lot of python codes in the package we are able to use. The code `launcher.py` contains the functions calling GPU or CPU computations. Currently, only Terachem and Turbomole computations are available. Gaussian or Qchem launcher might be added in the future.\
The program `candidate.py` contains functions that write Terachem input file. For now, it contains CASSCF, FOMO-CASCI, CASDFT, hh-TDA and hh-TDA-FOMO. This function does not contain some input parameters which are essential for computation (those input parameters are adjusted in the `test_input.yaml`, but we can still edit the convergence parameters inside. We are working on extrenalizing them into the yaml shortly.\
The `test_input.yaml` file contains information of active space (CASSCF, FOMO-CASCI), temperature (FOMO-CASCI, hh-TDA-FOMO), and width (hh-TDA, hh-TDA-FOMO). We need to input those parameters for different computations.\
The program `make_sh.py` Allows you to set the submission scripts so that it can be run on any machine. This program directly changes the `sys_utils.py` which controls the duplicating of submission scripts and execution of jobs.\
The program `autopilot.py` launches the Terachem and Turbomole computations. When running this, please type `python autopilot.py -i test_input.yaml`. This will automatically generate and launch a bunch of computations on GPU nodes, while writing one folder for each.\
The program `gradient.py` will run gradient calcuations from the candidates that ran through `autopilot.py` sucessfully, gradient time is used in the grader as it is a better indicator of candidate cost than the energies of n states.\
The program `spectra_grader.py` plots and grades each results automatically. Run with `python spectra_grader.py -i test_input.yaml`.
### Step-by-Step workflow
I recommend running calculations in a directory seperate from your autopilot directory to keep things clean and organinzed. As a result I will mention /path/to/autopilot/directory/ as a stand-in for your compiled autopilot path.\
0. Create the envirnoment if you haven't done so already: `conda env create -f environment.yml`\
1. Activate Autopilot: In terminal, type: `ml anaconda`, then `conda activate autopilot`. You may elect to optimize S0 isdie of autopilot first. To do this, please see `test_input_with_opt.yaml` for setup\
2. Set your submission scripts to work with your machine: `python /path/to/autopilot/directory/make_sh.py candidate_sh.txt ref_sh.txt`. Example `candidate_sh.txt` and `ref_sh.txt` are provided. You only need to do this once every time you need to use differnt settings on your machine\ 
3. Run the autopilot program: `python /path/to/autopilot/directory/autopilot.py -i test_input.yaml`\
4. Once autopilot calulations are done: Run gradient.py: `python /path/to/autopilot/directory/gradient.py -i test_input.yaml`\
5. Once gradient calulations are complete and `gradient.py` is done: Run the grader and collect results: `python /path/to/autopilot/directory/spectra_grader.py -i test_input.yaml`\

###Helpful Tips
If you would like to still have access to your terminal command line while gradient.py is running, put the command into a submission script in your machine when available. I personally do this for each step as a safe practice. 

## Visualizing your results: 
Organized scores will be in `Final_Scores.txt`.\
The plot with state energies and grader scores can be found in `S0min_results.png`.
`spectra_grader.py` will generate a directory `S0min_UVVis_plots` directory: here the plot represent the aligments of each state per candidate against the reference of same said candidate.\ Area Under the Curve (AUC) score is a measure of how accurate the shape of the spectra is in both absorptivity and relative energies of states. Min-Max normalization of average AUC gives AUC score.\ Min-Max normalization of the gradient run times of each candidate yields Time score, currently you can tune the weght of the time score, the default is even weight (1x multiplier).     
