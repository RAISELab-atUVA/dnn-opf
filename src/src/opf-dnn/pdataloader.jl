import Glob

include("datautils.jl")
include("utils.jl")

args = parse_commandline()
args["netname"] = "nesta_case1397sp_eir"


mutable struct PDataLoader
    value_pairs::Array
    experiments::Dict
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
    n::Int
    current_index::Int
    data_path::String
end

function PDataLoader(args, shuffle=false)
    path = "data/traindata/" * args["netname"] * "/"
    files = [basename(f) for f in Glob.glob("*.json", path)]
    # sort files and index them
    # TODO: Current implementation admit only 1 sample per \mu -- extend it to
    # multiple samples per mu
    experiments = Dict(parse(Float64, splitext(f)[1]) => f for f in files)
    _values = sort(collect(values(experiments)))
    scale = (args["state-distance"] / 100.0)

    value_pairs = []
    for x in _values
        _vals = [(x, y) for y in _values if (abs(y - x) <= scale) && (x != y)]
        value_pairs = vcat(value_pairs, _vals)
    end

    if isnothing(args["traindata-size"])
        n = length(value_pairs
    else
        n = min(length(value_pairs), args["traindata-size"])
    end
    train_split = args["split"] * 100
    train_idx = sample(1:n, Int(round(train_split * n)), replace = false)
    test_idx = setdiff(collect(1:n), train_idx)
    ntrain = length(train_idx)
    ntest  = length(test_idx)
    shuffle && randperm!(train_idx)

    return (PDataLoader(value_pairs,, experiments, args["batchsize"], shuffle, train_idx, ntrain, [], path),
        PDataLoader(value_pairs,, experiments, args["batchsize"],  shuffle, test_idx, ntest, [], path))
end


function Base.iterate(it::PDataLoader, start=1)
    if start > it.n
        it.shuffle && randperm!(it.indices)
        return nothing
    end
    nextstart = min(start + it.batchsize, it.n + 1)
    i = it.indices[start:nextstart-1]
    # save current indices
    it.current_indices = i

    data1 = JSON.parsefile(it.path * it._experiments[_value_pairs[1][1]])["experiments"][1])
    data2 = JSON.parsefile(it.path * it._experiments[_value_pairs[1][2]])["experiments"][1])

    v1, v2, Δv = Dict(), Dict(), Dict()
    for k in ["pd", "qd", "pg", "qg", "vm", "va", "pf", "qf", "pt", "qt"]
        v1[k] = collect(values(data1[k]))
        v2[k] = collect(values(data2[k]))
        v1[k][isnothing.(v1[k])] .= 0
        v2[k][isnothing.(v2[k])] .= 0
    end
    for k in ["vm", "va", "pg", "qg"]
        Δv[k] = v2[k] .- v1[k]
    end

    Sd1 = vcat(v1["pd"], v1["qd"])
    Sd2 = vcat(v2["pd"], v2["qd"])
    Flows1 = vcat(v1["pf"], v1["qf"], v1["pt"], v1["qt"])
    Flows2 = vcat(v2["pf"], v2["qf"], v2["pt"], v2["qt"])

    element = (vcat(Sd1, Sd2), v1["vm"], v1["va"], v1["pg"], Flows2, v2["vm"], vm["va"], v2["pg"])

    # element = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.dataset)
    # element = Tuple(copy(view(x, i, :)) for x in it.dataset)
    return element, nextstart
end

Base.length(it::PDataLoader) = it.n
Base.eltype(it::PDataLoader) = typeof(it.dataset)

# function batchselect(x::AbstractArray, i)
#     inds = CartesianIndices(size(x)[1:end-1])
#     x[inds, i]
# end

function Base.show(io::IO, it::PDataLoader)
    print(io, "DataLoader(dataset size = $(it.n)")
    print(io, ", batchsize = $(it.batchsize), shuffle = $(it.shuffle)")
    print(io, ")")
end



#(Sd, Flows, vm, va, pg, qg)
