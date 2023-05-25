using PyPlot
using Printf, ProgressMeter, Statistics

import JSON

include("graphs/graphs.jl")
include("opf-dnn/utils.jl")
include("opf-dnn/datautils.jl")
include("opf-dnn/dataloader.jl")
include("opf-dnn/pypickle.jl")

np    = pyimport("numpy")
random = pyimport("random")
plt = pyimport("matplotlib.pyplot")
pad = pyimport("pandas")
skl_t = pyimport("sklearn.tree")
skl_m = pyimport("sklearn.model_selection")

#=
args = parse_commandline()
name_train = args["netname"]
batch = args["batchsize"]
lr = args["lr"]
nepoc = args["nepochs"]
["nesta_case14_ieee",  "nesta_case30_ieee", "nesta_case39_epri",
    "nesta_case57_ieee", "nesta_case73_ieee_rts", "nesta_case89_pegase", "nesta_case118_ieee",
    "nesta_case162_ieee_dtc", "nesta_case189_edin", "nesta_case300_ieee"]

=#
for name_train in ["nesta_case14_ieee",  "nesta_case30_ieee", "nesta_case39_epri",
    "nesta_case57_ieee", "nesta_case73_ieee_rts", "nesta_case89_pegase", "nesta_case118_ieee",
    "nesta_case162_ieee_dtc", "nesta_case189_edin", "nesta_case300_ieee"]

    file_dt = "data/predictions/" * name_train * "/dt-results_fixed.pkl"
    pic_new = read_pickle(file_dt)
    print("$name_train: " )
    println(mean(pic_new["max_depth"]))
end