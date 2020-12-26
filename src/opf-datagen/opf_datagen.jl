using JuMP, Ipopt, PowerModels
using Random, Distributions
using JSON
using ProgressMeter
using ArgParse
PowerModels.silence() # suppress warning and info messages
Random.seed!(123)

""" Parse Arguments """
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "nesta_case14_ieee"
        "--output", "-o"
            help = "the output name"
            arg_type = String
            default = "traindata_ext"
        "--lb"
            help = "The lb (in %) of the load interval"
            arg_type = Float64
            default = 0.8
        "--ub"
            help = "The ub (in %) of the load interval"
            arg_type = Float64
            default = 1.2
        "--step"
            help = "The step size resulting in a new load x + step"
            arg_type = Float64
            default = 0.1
        "--nperm"
            help = "The number of load permutations for each laod scale"
            arg_type = Int
            default = 10
    end
    return parse_args(s)
end

function scale_load(data, scale_coef)
   newdata = deepcopy(data)
   for (i, (k, ld)) in enumerate(newdata["load"])
        if (ld["pd"] > 0)
            ld["pd"] = ld["pd"] * scale_coef[i]
            ld["qd"] = ld["qd"] * scale_coef[i]
        end
   end
   return newdata
end

function get_load_coefficients(µ, ∑, n)
    x = rand(TruncatedNormal(µ, ∑, µ-0.1, µ+0.1), n)
    model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    @variable(model, 0 <= _x[1:n] <= µ)
    @objective(model, Min, sum((_x[i] - x[i])^2 for i in 1:n))
    @constraint(model, mean(_x) == µ)
    JuMP.optimize!(model)
    return [JuMP.value(_x[i]) for i in 1:n]
end


args = parse_commandline()

data_path = "data/"
outdir = data_path * "traindata/" * args["netname"]
fileout = outdir * "/" * args["output"]  * ".json"
mkpath(outdir)
filein = data_path * "inputs/" * args["netname"] * ".m"
data = PowerModels.parse_file(filein)
Load_range = collect(args["lb"]:args["step"]:(args["ub"]))
solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)
nloads = length(data["load"])
res_stack = []


################
# Run tests
n = length(Load_range * args["nperm"])
p = Progress(n, 0.1)   # minimum update interval: 1 second
for µ in Load_range
    ∑ = 0.01

    for rep in 1:args["nperm"]
        load_scale = get_load_coefficients(µ, ∑, nloads)
        newdata = scale_load(data, load_scale)
        opf_sol = PowerModels.run_ac_opf(newdata, solver, setting = Dict("output" => Dict("branch_flows" => true)))

        if opf_sol["termination_status"] == LOCALLY_SOLVED
            # Retrieve: (p^d, q^d) and (p^g, v)
            res  = Dict{String, Any}()
            res["scale"] = mean(load_scale)
            res["pd"] = Dict(name => load["pd"] for (name, load) in newdata["load"])
            res["qd"] = Dict(name => load["qd"] for (name, load) in newdata["load"])
            res["vg"] = Dict(name => opf_sol["solution"]["bus"][string(gen["gen_bus"])]["vm"]
                                        for (name, gen) in newdata["gen"]
                                            if data["gen"][name]["pmax"] > 0)
            res["pg"] = Dict(name => gen["pg"] for (name, gen) in opf_sol["solution"]["gen"]
                                            if data["gen"][name]["pmax"] > 0)
            res["qg"] = Dict(name => gen["qg"] for (name, gen) in opf_sol["solution"]["gen"]
                                            if data["gen"][name]["pmax"] > 0)

            # Lines
            res["pt"] = Dict(name => data["pt"] for (name, data) in opf_sol["solution"]["branch"])
            res["pf"] = Dict(name => data["pf"] for (name, data) in opf_sol["solution"]["branch"])
            res["qt"] = Dict(name => data["qt"] for (name, data) in opf_sol["solution"]["branch"])
            res["qf"] = Dict(name => data["qf"] for (name, data) in opf_sol["solution"]["branch"])

            # Buses
            res["va"] = Dict(name => data["va"] for (name, data) in opf_sol["solution"]["bus"])
            res["vm"] = Dict(name => data["vm"] for (name, data) in opf_sol["solution"]["bus"])
            res["objective"] = opf_sol["objective"]
            res["solve_time"] = opf_sol["solve_time"]
            push!(res_stack, res)
        end
        next!(p)
    end
end

#########################
# Problem Constraints
pglim = Dict{}(name => (gen["pmin"], gen["pmax"]) for (name, gen) in data["gen"]
                                                    if data["gen"][name]["pmax"] > 0)
qglim = Dict{}(name => (gen["qmin"], gen["qmax"]) for (name, gen) in data["gen"]
                                                    if data["gen"][name]["pmax"] > 0)
vglim = Dict{}(name => (data["bus"][string(gen["gen_bus"])]["vmin"],
                       data["bus"][string(gen["gen_bus"])]["vmax"])
                          for (name, gen) in data["gen"]
                              if data["gen"][name]["pmax"] > 0)

vm_lim = Dict{}(name => (bus["vmin"], bus["vmax"]) for (name, bus) in data["bus"])
rate_a = Dict{}(name => (branch["rate_a"]) for (name, branch) in data["branch"])
                                                    # !gen["transformer"]
line_br_rx = Dict{}(name =>
        (branch["br_r"], branch["br_x"]) for (name, branch) in data["branch"])
line_bg = Dict{}(name =>
        (branch["g_to"] + branch["g_fr"],
         branch["b_to"] + branch["b_fr"]) for (name, branch) in data["branch"])

out_res = Dict{String, Any}()
out_res["experiments"] = res_stack
out_res["constraints"] = Dict("vg_lim" => vglim, "pg_lim" => pglim, "qg_lim" => qglim,
                              "vm_lim" => vm_lim,
                              "rate_a" => rate_a, "line_rx" => line_br_rx,
                              "line_bg" => line_bg)

##########################
# Write file out
open(fileout, "w") do f
    write(f, "$(JSON.json(out_res, 4))")
    println("Writing:  ", fileout)
end
