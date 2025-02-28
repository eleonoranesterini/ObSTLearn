# PRE-REQUISITES:

pip3 install numpy

pip3 install z3-solver

pip3 install pandas


Install RTAMT from the branch weak-next: 

git clone --branch weak-next https://github.com/nickovic/rtamt.git

cd rtamt/

pip3 install .


(To avoid warning messages printed)
pip install antlr4-python3-runtime==4.5


# RUN STL-MINING FOR LEAD-FOLLOWER:

python experiments/lead_follower/experiment_lead_follower.py


# RUN STL-MINING FOR LEAD-FOLLOWER:

python experiments/traffic_cones/experiment_traffic_cones.py


# TEST THE MINED FORMULA ON THE TESTING SET AND COMPUTE PERFORMANCE METRICS

Copy the folder(s) containing the (set of) ensemble(s) inside the folder "ensemble_to_be_tested_in_here"

and run

python script_testing.py
