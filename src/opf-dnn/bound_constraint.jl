include("torch.jl")

"""
    Strut to encode Bound constraints.
"""
mutable struct BoundConstraint
    lb::Any
    ub::Any
    zeros::Any
    is_abs::Bool

    function BoundConstraint(data, config, type, device=nothing)
        if device == nothing
            device = torch.device("cpu")
        end
        BS = config["batchsize"]

        @assert type in ["pg-bnd", "qg-bnd", "vg-bnd", "vm-bnd", "va-bnd", "flow-bnd"]
        _type = type[1:findfirst("-bnd", type)[1]-1]

        if _type == "flow"
            x_lim = hcat(collect(values(data["constraints"]["rate_a"]))...)'
            x_lim = x_lim.^2
            x_min = nothing
            x_max = Ten(repeat(x_lim'; outer=BS)).to(device)
            is_abs = true
        elseif _type == "va"
            n_buses = length(data["experiments"][1]["va"])
            # x_min, x_max = -pi/6, pi/6
            x_min = torch.zeros(BS, n_buses).fill_(-pi/6).to(device)
            x_max = torch.zeros(BS, n_buses).fill_(pi/6).to(device)
            is_abs = false
        else
            x_lim = hcat(collect(values(data["constraints"][_type*"_lim"]))...)'
            # x_min, x_max = Ten(selectdim(x_lim, 2, 1)), Ten(selectdim(x_lim, 2, 2))
            x_min, x_max = selectdim(x_lim, 2, 1), selectdim(x_lim, 2, 2)
            x_min = Ten(repeat(x_min'; outer=BS)).to(device)
            x_max = Ten(repeat(x_max'; outer=BS)).to(device)
            is_abs = false
        end
        zeros = torch.zeros_like(x_max).to(device)
        new(x_min, x_max, zeros, is_abs)
    end
end

function get_bounds(bc::BoundConstraint, x)
    _lb, _ub, _z = bc.lb, bc.ub, bc.zeros
    if x.shape[1] != bc.ub.shape[1]
        _lb = bc.is_abs ? bc.lb : bc.lb.narrow(0, 0, x.shape[1])
        _ub = bc.ub.narrow(0, 0, x.shape[1])
        _z = bc.zeros.narrow(0, 0, x.shape[1])
    end
    return _lb, _ub, _z
end

# Loss functions on a vector of variables bounds
function bound_penalty(bc::BoundConstraint, x)
    lb, ub, z = get_bounds(bc, x)
    if ! bc.is_abs
        return torch.mean(torch.max(z, (lb - x)))
               + torch.mean(torch.max(z, (x - ub)))
    else
        return torch.mean(torch.max(z, (x - ub)))
    end
end

function bclamp(bc::BoundConstraint, x)
    lb, ub, z = get_bounds(bc, x)
    if ! bc.is_abs
        return torch.max(torch.min(x, ub), lb)
    else
        return torch.min(x, ub)
    end
end
