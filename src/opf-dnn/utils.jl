using Printf, ArgParse, Statistics, ProgressMeter
include("torch.jl")

""" Parse Arguments """
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netpath" # do not change
            help = "The path to the input networks (.m)"
            arg_type = String
            default = "data/inputs/"
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "nesta_case14_ieee"
        "--traindata", "-i" # can read pickle files
            help = "The name of the input file, within the netname folder"
            default = "traindata.json"
        "--out-suffix", "-s" ## also used for version
            help = "The suffix given to the output file to identify a given program variant"
            default = nothing
        "--plot-outfile", "-p"
            help = "The name of the result file, within the netname folder"
            default = "losses.png"
        "--results-outfile", "-r"
            help = "The name of the result file, within the netname folder"
            default = "results.pkl"
        "--use-state"
            help = "Exploit hot-start state"
            default = false
            action = :store_true
        "--use-constraints"
            help = "Use Lagrangian constraint penalties"
            default = true
            action = :store_true
        "--use-dual-update"
            help = "Use Lagrangian dual update"
            default = false
            action = :store_true
        "--nocuda"
            help = "Do not use CUDA"
            action = :store_true
        "--verbose"
            action = :store_true
        "--nettype", "-t"
            help = "enc[oder] or dec[oder]"
            arg_type = String
            default = "dec"
        "--nepochs", "-e"
            help = "The number of epochs"
            arg_type = Int
            default = 10
        "--batchsize", "-b"
            help = "The size of the batch"
            arg_type = Int
            default = 10
        "--state-distance"
            help = "The distance, in percentage, between each two network
                    states, used in the construction of hot-start states
                    training data"
            arg_type = Float64
            default = 1.0 # = 1 %
        "--seed"
            arg_type = Int
            default = 1234
        "--split"
            help = "Train split in (0, 1). The rest is given to Test"
            arg_type = Float64
            default = 0.8
        "--lr"
            help = "The learning rate"
            default = 0.001
        "--dur"
            help = "Dual update rate"
            default = 0.01
        "--traindata-size"
            help = "maxmium traindata size"
            default = 100000
    end
    return parse_args(s)
end

""" Sum a vector of numbers if the vector is non empty, otherwise returns 0 """
function sum0(vec)
    return length(vec) > 0 ? sum(vec) : 0.0
end

""" Backprop function """
function backprop(agent, loss, retain_graph=false)
    agent.optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    agent.optimizer.step()
end

"""" Return pd[t-1], pd[t], qd[t-1], qd[t] for setpoint test """
function get_combo_Sd(x, nloads, numpy)
    if numpy
        return x.narrow(1, 0, nloads).cpu().numpy(),          # pd[t-1]
               x.narrow(1, nloads, nloads).cpu().numpy(),     # pd[t]
               x.narrow(1, 2*nloads, nloads).cpu().numpy(),   # qd[t-1]
               x.narrow(1, 3*nloads, nloads).cpu().numpy()    # qd[t]
    else
        return x.narrow(1, 0, nloads),          # pd[t-1]
               x.narrow(1, nloads, nloads),     # pd[t]
               x.narrow(1, 2*nloads, nloads),   # qd[t-1]
               x.narrow(1, 3*nloads, nloads)    # qd[t]
    end
end

""" Given a file name (may include its path), returns its extension """
extension(url::String) = try match(r"\.[A-Za-z0-9]+$", url).match catch nothing end

""" Given a file name (may include its path), returns its name """
filename(url::String) = try match(r"([A-Za-z0-9]+-*+_*)*", url).match catch nothing end

function get_file_name(config, type, prefix=nothing)
    @assert type in ["results", "model", "plot"]

    _PATH = "data/predictions/" * config["netname"] * "/"
    _nettype = args["nettype"] * "-"
    _suffix = config["out-suffix"] == nothing ? "" : "-" * config["out-suffix"]
    _prefix = prefix == nothing ? "" : "-" * prefix

    if type == "results"
        _fname, _ext = filename(config["results-outfile"]), extension(config["results-outfile"])
    elseif type == "plot"
        _fname, _ext = filename(config["plot-outfile"]), extension(config["plot-outfile"])
    elseif type == "model"
        _fname, _ext = "model", ".mdl"
    end

    return _PATH * _nettype * _fname * _prefix * _suffix * _ext
end
