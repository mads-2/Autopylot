# autocas
### My test for this stuff
First, we need to create environment for autocas. Type the command: `conda env create -f environment.yml`.\
After that, before running the package, we need to activate autopilot environment on cluster: `ml anaconda`, following by `conda activate autopilot`.\
There are a lot of python codes in the package we are able to use. The code `launcher.py` contains the functions calling GPU or CPU computations. Currently, only Terachem and Turbomole computations are available. Gaussian or Qchem launcher might be added in the future.\
The program `candidate.py` contains functions that write Terachem input file. For now, it contains CASSCF, FOMO-CASCI, CASDFT, hh-TDA and hh-TDA-FOMO. This function does not contain some input parameters which are essential for computation (those input parameters are adjusted in the `test_input.yaml`, but we can still edit the convergence parameters inside.\
The `test_input.yaml` file contains information of active space (CASSCF, FOMO-CASCI), temperature (FOMO-CASCI, hh-TDA-FOMO), and width (hh-TDA). We need to input those parameters for different computations.\
The program `autopilot.py` launches the Terachem and Turbomole computations. When running this, please type `python autopilot.py -i test_input.yaml`. This will automatically generate and launch a bunch of computations on GPU nodes, while writing one folder for each.\
The program `energies_parser.py` reads and extracts data from computation output files, while `results_collector.py` makes data into a plotable .csv file. You can test this by typing `python results_collector.py -i test_input.yaml`.\
The program `grader.py` plots and grades each results automatically. Run with `python grader.py -i test_input.yaml`. Note that when EOM results are not available, it will not be able to print but still able to plot. You might want to comment out those lines with EOM and still do the plotting.
