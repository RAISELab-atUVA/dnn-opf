# AC-OPF Learning
***Author***: Ferdinando Fioretto

***Last update***:  July, 9, 2019

# Usage
#### Dataset Generation
The training set for a network <netname> are generated using the following command:

```
  julia src/opf_solver/opf_datagen.jl --netname nesta_case14_ieee
                                      --lb 0.8 --ub 1.2 --step 0.0001
```

It will generate test cases varying the nominal power loads (pd, qd) by a
multiplicative factor $\delta \in [_lb_, _ub_]$ with a step size indicated by
the _step_ parameter.

#### OPF-training Step  
After the training data is generated we can train our Neural Network.
To do so, run the following command:

```
  julia src/ml-opf.jl --netname nesta_case14_ieee
```
The following arguments can be set:
- _--nocuda_  (Do not use CUDA).
  + Default value: false
- _--traindata_ The name of the input file, within the "netname" folder.
  It expects a file formatted as the output of the dataset generation procedure.
  + Default value: "traindata.json"
- _--outfile_ The name of the output file, within the "netname" folder.
  + Default value: "results.json"
- _--nepochs_ The number of epochs.
  + Default value: 10
- _--batchsize_ The size of the batch.
  + Default value: 1
- _--split_ Train split in (0, 1). The rest is given to Test.
  + Default value: 0.8
- _--version_ Problem version [1, 2].
    + version 1: outputs (**pg**, **vg**)
    + version 2: outputs (**pg**, **vg**), (S_ij, S_ji)
    + Default value: 2
- _--lr_ The learning rate.
  + Default value: 0.001
- _--c_pv_bnd_ Activate Lagrangian on Pg and Vg bounds.
  + Default value: true
- _--c_flow_loss_ Activate lagrangian on total flow loss.
  + Default value: false
- _--c_thermolimits_ Activate thermolimits loss
  + Default value: false
  + Version required: >= 2

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



#### OPF-training For Individual Components
To run the code:

```
  julia src/opf_classifier/unittests/test_comp_bounds/<PROGRAM>.jl
    --netname <NET-NAME>
    --nepochs 50
    --batchsize 10
    --type <TYPE>
    [--activation relu]
    [--loss mse]
    [--report-train-errors]
    [--report-verbose]
```
where PROGRAM is one of:
- learnPg_fromSd      -> type = "None"   
- learnQg_fromSd      -> type = "None"
- learnVa_fromSd      -> type = "None"
- learnVm_fromSd      -> type = "None"
- learnSfrom_fromSd   => type in ["tn", "ln", "both"]
- learnSto_fromSd     => type in ["tn", "ln", "both"]

The arguments _report-verbose_, and _report-train-errors_ if set, force the
program to report an output file that reports, respectively, the errors
in the predicted quantity for each test case analyzed, and the same analysis done
for the test-dataset but performed also on the train-dataset.


##### Result Output
The output will be stored in the path
> data/predictions/<NETNAME>/bnd_batch_<BATCHSIZE>/xxxx.json

The result file is a dictionary that contains the following
- **settings**: This is a test-case of the file settings
- **test**:    The errors recorded during _testing_ (L1 distance)
  + *mean-err* The average error
  + *min-err*  The minimum error
  + *max-err*  The maximum error
  + *sd-err*   The standard deviation over the errors recorded
  + **violations**
    + *average(-lb)* The average lower-bound errors
    + *average(-ub)* The average upper-bound errors
    + *mean(-lb)*    The normalized *number* of unsat test-cases due to lb violation
    + *mean(-ub)*    The normalized *number* of unsat test-cases due to ub violation
    + *total(-lb)*     The total number of constraint violations due to lb
    + *total(-ub)*     The total number of constraint violations due to ub
- **train**:   The training errors
  + same fields as for testing
- **losses**:  
  + *mse*   The loss vector generated during training. Every iteration contains
  a loss value.


# Scripts
The scripts in the "scripts" folder are used with condor.

- To generate the training data for all the networks run
```
  condor_submit inputs.submit
```
Then, run:
```
  python merge_data.py
```
to post-process the data (will merge multiple generated, split, files into one
  so it can be loaded by the training procedure)

- To train and test the networks, run:
```
  condor_submit run.submit
```
---

# AC-OPF Learning model
Inputs:
- vector of demands: (p^d, q^d) \in R^2n
Output:
- vector of generators and voltage levels:
  (p^g, v^g) \in R^2n

Once the output vector is given, the values for the vectors q^g, \delta can
be retrieved as solving a power flow problem.


# Results
<a name="results">

## Single Component Learning
- [Learning with MSE_loss](docs/results_1.md)
- [Learning with Smooth L1 Loss](docs/results_2.md)
- [Learning with Specialized Layer for Bound constraints](docs/results_3.md)

We use a fully connected NN with 3 hidden layers, each with ReLU activation :

Input         H1         H2         H3  Output
  [n] -> [n * 2] -> [n * 2] -> [m * 2]  -> [m]



### TODO
#### Debug
- [ ] Debug iterator in data-loader (make sure index retrieved are those corresponding to the data)

- Study which test cases (multiplicative factor of load) cause:
  + Constraint unsat
  + Large deviations from original values

- Weights for loss functions
  + [x] Include loss function plot recording
- Make activation function on last layer that satisfies bounds!
  - [x] clamp [lb, ub]

### Notes:
#### Input/Output
- Inputs:
	+ ["experiments"][i]["pd"], ["experiments"][i]["qd"]
- Outputs
    + ["experiments"][i]["pg"], ["experiments"][i]["vm"]
- Constraints
	+ ["constraints"]["vlim"][i] <- (min, max)
	+ ["constraints"]["plim"][i] <- (min, max)

#### Experiments
- Using 80/20 train/test split

### Change-Log:
#### v1
- Loss function:
  + Minimize the L2 of || pg - pg' || + || vm - vm'||

- Penalizers:
  + pg' > p_min  ::  if pg' < p_min =>  penalize as (pg' - p_min)^2
  + pg' < p_max  ::  if pg' > p_max =>  penalize as (pg' - p_max)^2
  + vm' > v_min  ::  if vm' < v_min =>  penalize as (vm' - v_min)^2
  + vm' < v_max  ::  if vm' > v_max =>  penalize as (vm' - v_max)^2

I've tried to remove the penalizer and the verification score significantly lower!

#### v2
Add the following to v1
- Loss function
  + Model power line losses as: \sum p^g - \sum p^d - \sum losses
  where the losses are defined as (\sum p^t + \sum p^f)

The best version has:
- c_pv_bnd true
- c_flow_loss false
- c_thermolimits false
- c_flow_mse true


### Links
- Different Losses for Different Layers
  + https://discuss.pytorch.org/t/how-to-implement-a-deep-neural-network-with-different-losses-for-different-layers/13851
  + https://discuss.pytorch.org/t/multiple-output-tutorial-examples/3050
  + https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
  + https://towardsdatascience.com/analyzing-different-types-of-activation-functions-in-neural-networks-which-one-to-prefer-e11649256209
  + https://github.com/torch/nn/blob/master/doc/simple.md#nn.IndexLinear
  + https://www.google.com/search?q=pytorch+layer+for+adding+two+tensors&oq=pytorch+layer+for+adding+two+tensors&aqs=chrome..69i57j33.5583j0j7&sourceid=chrome&ie=UTF-8

- Extending and Customizing layers
  + https://pytorch.org/docs/stable/notes/extending.html
  + https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa


- Sharing networks parameters (freezing / unfreezing)
  + https://www.datascience.com/blog/transfer-learning-in-pytorch-part-two
  + https://discuss.pytorch.org/t/correct-way-to-freeze-layers/26714/4
  + https://discuss.pytorch.org/t/how-to-freeze-a-specific-layer-in-pytorch/14868
