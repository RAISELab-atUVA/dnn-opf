using Random
using StatsBase, MLBase

using DelimitedFiles

"""
    DataLoader(dataset...; batchsize::Int=100, shuffle=true)
DataLoader provides iterators over the dataset.
```julia
X = rand(10, 1000)
Y = rand(1, 1000)
m = Dense(10, 1)
loss(x, y) = Flux.mse(m(x), y)
opt = ADAM(params(m))
trainloader = DataLoader(X, Y, batchsize=256, shuffle=true)
Flux.train!(loss, trainloader, opt)
```
"""
mutable struct DataLoader
    dataset::Tuple
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
    n::Int
    current_indices::Vector{Int}
end

function DataLoader(dataset::NTuple{N,AbstractArray}; batchsize::Int=100, shuffle, train_split=0.8, kfold_it) where N

    f = first.(size.(dataset))  # number of elements in each tuple of dataset
    n = first(f)
    l = last.(size.(dataset))   # number of features in each tuple of dataset
    all(n .== f) || throw(DimensionMismatch("All data should have the same length."))
    
    # Take seed from Args
    rng = TaskLocalRNG()
    
    idx = collect(1:n)
    idx1 = idx
    shuffle && shuffle!(rng,idx) # Shuffle entire dataset

    dataset_idx = Any[]

    # Prepare 5 arrays of idx
    for i in 0:4
        push!(dataset_idx, idx[i*n÷5+1:(i+1)*n÷5])
    end
    #Assign data to train, eval and test set
    train_idx = vcat(dataset_idx[kfold_it], dataset_idx[mod(kfold_it+1,1:5)],
    dataset_idx[mod(kfold_it+2,1:5)])
    val_idx = dataset_idx[mod(kfold_it+3,1:5)]            
    test_idx = dataset_idx[mod(kfold_it+4,1:5)]

    ntrain = length(train_idx)
    ntest  = length(test_idx)
    nval  = length(val_idx)

    return (DataLoader(dataset, batchsize, shuffle, train_idx, ntrain, []),
            DataLoader(dataset, batchsize, shuffle, test_idx, ntest, []),
            DataLoader(dataset, batchsize, shuffle, val_idx, nval, []))
end

DataLoader(dataset...; batchsize::Int=100, shuffle=true, train_split=0.8, kfold_it=1) =
    DataLoader(dataset, batchsize=batchsize, shuffle=shuffle, train_split=0.8, kfold_it=kfold_it)

#=DataLoader(dataset...; batchsize::Int=100, shuffle=false, train_split=0.8) =
    DataLoader(dataset, batchsize=batchsize, shuffle=shuffle, train_split=0.8)=#

function Base.iterate(it::DataLoader, start=1)
    if start > it.n
        it.shuffle && randperm!(it.indices)
        return nothing
    end
    nextstart = min(start + it.batchsize, it.n + 1)
    i = it.indices[start:nextstart-1]
    # save current indices
    it.current_indices = i

    # element = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.dataset)
    element = Tuple(copy(view(x, i, :)) for x in it.dataset)
    return element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = typeof(it.dataset)

# function batchselect(x::AbstractArray, i)
#     inds = CartesianIndices(size(x)[1:end-1])
#     x[inds, i]
# end

function Base.show(io::IO, it::DataLoader)
    print(io, "DataLoader(dataset size = $(it.n)")
    print(io, ", batchsize = $(it.batchsize), shuffle = $(it.shuffle)")
    print(io, ")")
end
