# AC-OPF Learning
***Author***: Ferdinando Fioretto

***Last update***:  July, 9, 2019

This repository contains the code is associated with paper: 

_Predicting AC Optimal Power Flows: Combining Deep Learning and Lagrangian Dual Methods_. 
Ferdinando Fioretto, Terrence W.K. Mak, Pascal Van Hentenryck. 
In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2020.

**Cite as**

```bibtex
@inproceedings{Fioretto:AAAI20, 
  author    = {Ferdinando Fioretto and {Terrence W.K.} Mak and Pascal {Van Hentenryck}},
  title     = {Predicting {AC} Optimal Power Flows: Combining Deep Learning and Lagrangian Dual Methods},
  pages     = {630--637},
  publisher = {{AAAI} Press},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence {(AAAI)}}
  url    =  {https://ojs.aaai.org/index.php/AAAI/article/view/5403}, 
  DOI    =  {10.1609/aaai.v34i01.5403}, 
  year   = {2020}
}
```

# Usage

**Installation**
Julia Packages:
- PowerModels v0.13.2
- JuMP v0.21.8
- Ipopt v0.6.5
- Random
- Distributions
- JSON
- ProgressMeter v1.7.1
- ArgParse v1.1.4
- PyCall v1.92.3
- PyPlot

Python Packages
- torch
- numpy
- random

**Dataset Generation**
The training set for a network <netname> are generated using the following command:

```
  julia src/opf_solver/opf_datagen.jl --netname nesta_case14_ieee
			      	      --lb 0.8 --ub 1.2 --step 0.0001
```
It will generate test cases varying the nominal power loads (pd, qd) by a
multiplicative factor $\delta \in [_lb_, _ub_]$ with a step size indicated by
the _step_ parameter.
	
**OPF-training Step**
	
After the training data is generated we can train our Neural Network.
To do so, run the following command:

```
  julia src/opf-dnn.jl --netname nesta_case14_ieee
```

The following arguments can be set:
- _--netpath_  The path to the input networks (.m) (Do not change).
  + Default value: "data/inputs/"
- _--netname -n  The input network name.
  + Default value: nesta_case14_ieee
- _--traindata --i_ The name of the input file, within the "netname" folder.
  It expects a file formatted as the output of the dataset generation procedure.
  + Default value: "traindata.json"
- _--out-suffix -s_  he suffix given to the output file to identify a given program variant.
  + Default value: nothing
- _--plot-outfile -p_ The name of the result file, within the netname folder.
  + Default value: losses.png
- _--results-outfile_ The name of the result output file, within the "netname" folder.
  + Default value: "results.pkl"
- _--use-state"_  Exploit hot-start state.
  + Default value: false
- _--use-constraints"_  Use Lagrangian constraint penalties.
  + Default value: true
- _--use-dual-update_  Use Lagrangian dual update.
  + Default value: false
- _--nocuda_  Do not use CUDA.
  + Default value: false
- _--nettype -t_ Enc[oder] or dec[oder].
  + Default value: false
- _--nepochs -e_ The number of epochs.
  + Default value: 10
- _--batchsize -b_ The size of the batch.
  + Default value: 10
- _--split_ Train split in (0, 1). The rest is given to Test.
  + Default value: 0.8
- _--state-distance_ The distance, in percentage, between each two network
                    states, used in the construction of hot-start states
                    training data.
  + Default value: 1.0
- _--seed_ 
  + Default value: 1234
- _--lr_ The learning rate.
  + Default value: 0.001
- _--dur_ Dual update rate.
  + Default value: 0.01
- _--traindata-size_ Maxmium traindata size
  + Default value: 100000

The program above executes two main steps:

1. A Training Step: It trains the neural network and produces the vector (**pg**, **vg**)
  of predictions.

2. A Testing Step: This testing routine calls a _restoration_ program for
  each network in the test set. The _restoration_ program leaves a free slack bus,
  runs an OPF (i.e., it finds an assignment for the variables <**qg**, **va**, **vm**>).
  Recall that we predict the voltage magnitude only for generator buses.

The second step produces an output file, saved by default in "results.json". Its
component are reviewed next.

#### Result Output

To generate the desired results, set: write_loadflow_results, write_test_results, write_training_losses, write_summary_results to _true_ in opf-dnn.jl (r. 10-14).

The result file, output of the testing restoration phase is described by the
following components. All these results pertain the networks generated in the
_test set_:

- _settings_: It contains the parameters with which the training step was run.
  These include the arguments of the _ml-opf.jl_ program.
- _results_:
  + _n_fail_: The number of instances for which a restoration procedure failed.
  + _n_success_: The number of instances for which a restoration procedure succeeded.
  + _n_primal_feasible_: The number of instances for which a restoration procedure
  did not converged but a primal feasible solution is returned.
  + _avg_objective_diff_: The average OPF objective difference between that
  associated to the original networks and that obtained during the restoration
  phase using the predicted (**pg**, **vg**).
  + _avg_err_pg_: The average difference between the vectors **pg** in the original
  network and in the predictions.
  + _avg_err_vg_: The average difference between the vectors **vg** in the original
  network and in the predictions.
  + _avg_solve_time_restoration_: The average solve time taken by the restoration
  procedure.
  + _avg_solve_time_ori_: The average solve time taken to solve the original OPF
  problems.
  + _test_errors_: The errors obtained during the **learning test phase**. In this phase,
  we simply verify the prediction outputs of networks not seen during training.
    + _err_pv_: The MSE error for the vector predicted vectors **pg** and **vg**
    w.r.t. the original ones
    + _err_flow_: The MSE error for the vector predicted vector **p_from**,
    **q_from**, **p_to**, **q_to** w.r.t. the original ones
  + _train_losses_: The vector of losses obtained during the training step. We
  store the losses obtained at each training iteration, for each loss function
  adopted.
    + _mse_pv_
    + _mse_flow_ (only for ver >= v2)
    + _vg_bnd_ (if _c_pv_bnd_ constraint is active)
    + _pg_bnd_ (if _c_pv_bnd_ constraint is active)
    + _thermo_ (if _c_thermolimits_ constraint is active, and ver >= 2)
    + _real_flow_loss_ (if _c_flow_loss_ constraint is active)

# AC-OPF Learning model
	
Inputs:
- vector of demands: $(p^d, q^d) \in \mathbb{R}^{2n}$
Output:
- vector of generators and voltage levels:
  $(p^g, v^g) \in \matbb{R}^{2n}$

Once the output vector is given, the values for the vectors $q^g, \delta$ can be retrieved as solving a power flow problem.
