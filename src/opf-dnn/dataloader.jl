using Random:randperm!
using StatsBase

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

function DataLoader(dataset::NTuple{N,AbstractArray}; batchsize::Int=100, shuffle=false, train_split=0.8) where N
    #l = last.(size.(dataset))
    f = first.(size.(dataset))  # number of elements in each tuple of dataset
    n = first(f)
    l = last.(size.(dataset))   # number of features in each tuple of dataset
    all(n .== f) || throw(DimensionMismatch("All data should have the same length."))

    train_idx = sample(1:n, Int(round(train_split * n)), replace = false)
    test_idx = setdiff(collect(1:n), train_idx)
    ntrain = length(train_idx)
    ntest  = length(test_idx)
    shuffle && randperm!(train_idx)

    return (DataLoader(dataset, batchsize, shuffle, train_idx, ntrain, []),
        DataLoader(dataset, batchsize, shuffle, test_idx, ntest, []))
end

DataLoader(dataset...; batchsize::Int=100, shuffle=false, train_split=0.8) =
    DataLoader(dataset, batchsize=batchsize, shuffle=shuffle, train_split=0.8)

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
