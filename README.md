# apl-rram-nn-codes

This repository contains the codes utilized for our publication submitted to APL-ML. We will add further information as soon as the publication has gone through review.

## Running the code

### Installing requried packages

The following only needs to be performed once:

* Set up a Python virtual environment using `python -m venv aplenv`
* Switch to the created environment using `source aplenv/bin/activate`
* install the required packages using `pip install -r requirements.txt`

The corresponding environment can be closed by typing `deactivate`

### Running the training procedure

The flow is described in the `do.sh` script which can be run as follows:

* activate the environment by using `source aplenv/bin/activate`
* run the bash script by typing `./do.sh`. 

This will train the three diffrently sized networks by selecting the to-be-omitted fraction of the utilized dataset using a commandline parameter. This will train 20 networks in parallel, this might need to be adapted to your machine. (20 networks in parallel is fine for a dual-socket server equipped with Epyc 7713). A single individual network can be trained by typing

`python ML_test_l.py n`

where n should be an integer between 1 and 100. Specifying `n=100` will use the maximum fraction allowed in the current configuration. This configuration can be adapted in line 21 of the corresponding python file.

