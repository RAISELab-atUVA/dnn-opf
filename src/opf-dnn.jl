"""
	OPF-DNN
	Author: Ferdinando Fioretto
	Date:   Sept. 1, 2019

	Reference: Ferdinando Fioretto, Terrence W. K. Mak, Pascal Van Hentenryck:
		 Predicting AC Optimal Power Flows: Combining Deep Learning and
		 Lagrangian Dual Methods. CoRR abs/1909.10461 (2019)
"""
write_loadflow_results = true
write_test_results     = false
write_training_losses  = false
write_summary_results  = false

include("opf-dnn/utils.jl")

# Parse Arguments
args = parse_commandline()
fix_random_params(args["seed"])

if args["use-state"]
	include("opf-dnn/agent_S.jl")
else
	include("opf-dnn/agent_N.jl")
end

# Create agent and Train it
agent = OpfAgent(args)
losses = train(agent)

# Save Model with weights
save_model(agent, args)

#########
# Save Results and post-process
########
include("opf-datagen/restoration_w_hotstart.jl")
include("opf-dnn/report.jl")

# Modify Test data
exp = read_data(args, args["traindata"])["experiments"]
scales = collect(0.8:0.01:1.2)
# (index of train data, index of the previous state)
prev_state_idx  = [(t, agent.data_indexes[t, 1]) for t in agent.test_loader.indices]
scales_idx = Dict(s=>[] for s in scales)
for (tix, sidx) in prev_state_idx
	s = exp[sidx]["scale"]
	if !args["use-state"]
		s = round(s, digits=2)
	end

	if s in scales
		push!(scales_idx[s], tix)
	end
end

# Reduce Test set if using the previous state
if args["use-state"]
	new_indexes = vcat(collect(values(scales_idx))...)
	agent.test_loader.indices = new_indexes
	agent.test_loader.n = length(new_indexes)
end

# Testing
if (write_test_results || write_loadflow_results)
	predictions, errors = test(agent)
end

results = Dict()
results["scales"] = scales_idx

if write_test_results
	results["test_errors"] = errors
	results["predictions"] = predictions
end

## Solve Load-Flow Problem (slow)
if write_loadflow_results
	results["loadflow"] = solve_restoration_problem(args, predictions, agent.data_indexes)
end

## Result summary
if write_summary_results
  write_data(Dict("vm"=>mean(errors["vm"]),
                  "va"=>mean(errors["va"]),
                  "pg"=>mean(errors["pg"]),
				  #"qg"=>mean(errors["qg"]),
                  "ohm"=>mean(errors["ohm"]),
                  "klc"=>mean(errors["klc"])),
             args, "summary")
end

if write_training_losses
	save_plot_losses(losses, args, "far")
	save_plot_losses(losses, args, "reg", 0.1)
	save_plot_losses(losses, args, "zoom", 0.01)
	results["train_losses"] = losses
end

if (write_test_results || write_loadflow_results || write_training_losses || write_summary_results)
	write_data(Dict("results" => results, "settings" => args), agent.config)
end
