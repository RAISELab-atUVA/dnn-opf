include("torch.jl")

"""
    Struct to encode Ohms Law constraints.
    Given (predicted) voltage angle and voltage magnitude, it is used to compute
    the associated active and load flow (in both directions).
"""
mutable struct OhmsLawConstraint
    # indexes of from buses and to buses for each line (ordered according to br_names)
    device::Any
    idx_bus_fr::Any
    idx_bus_to::Any
    tm::Any
    ta::Any
    # vector of G and B for each line
    g::Any
    b::Any
    p_fr1::Any
    q_fr1::Any
    p_to1::Any
    q_to1::Any

    function OhmsLawConstraint(datainfo, config, device=nothing)
        if device == nothing
            device = torch.device("cpu")
        end
        pnet = datainfo.powernet
        BS = config["batchsize"]

        br_names = collect(datainfo.keys["pf"])
        ori_branch = Dict(k => pnet["branch"][k] for k in br_names)
        f_bus = [ori_branch[k]["f_bus"] for k in br_names]'
        t_bus = [ori_branch[k]["t_bus"] for k in br_names]'
        _gb = [calc_branch_y(ori_branch[k]) for k in br_names]
        g = [_gb[i][1] for i in 1:length(_gb)]'
        b = [_gb[i][2] for i in 1:length(_gb)]'

        g_fr = [ori_branch[k]["g_fr"] for k in br_names]'
        b_fr = [ori_branch[k]["b_fr"] for k in br_names]'
        g_to = [ori_branch[k]["g_to"] for k in br_names]'
        b_to = [ori_branch[k]["b_to"] for k in br_names]'
        tm = [ori_branch[k]["tap"] for k in br_names]'
        ta = [ori_branch[k]["shift"] for k in br_names]'
        idx_fr = [findall(x->x==string(f_bus[i]), datainfo.keys["vm"])[1] for i in 1:length(f_bus)]
        idx_to = [findall(x->x==string(t_bus[i]), datainfo.keys["vm"])[1] for i in 1:length(t_bus)]

        idx_bus_fr = iTen(idx_fr.-1).to(device)
        idx_bus_to = iTen(idx_to.-1).to(device)
        _p_fr_1 = Ten(repeat((g .+ g_fr); outer=BS)).to(device)
        _q_fr_1 = Ten(repeat(-(b .+ b_fr); outer=BS)).to(device)
        _p_to_1 = Ten(repeat((g .+ g_to); outer=BS)).to(device)
        _q_to_1 = Ten(repeat(-(b .+ b_to); outer=BS)).to(device)
        tm = Ten(repeat(tm; outer=BS)).to(device)
        ta = Ten(repeat(ta; outer=BS)).to(device)
        g = Ten(repeat(g; outer=BS)).to(device)
        b = Ten(repeat(b; outer=BS)).to(device)

        # _p_fr_1 = Ten(g .+ g_fr ).to(device)
        # _q_fr_1 = Ten(-b .+ b_fr).to(device)
        # _p_to_1 = Ten(g .+ g_to).to(device)
        # _q_to_1 = Ten(-b .+ b_to).to(device)

        new(device, idx_bus_fr, idx_bus_to, tm, ta,
            g, b, _p_fr_1, _q_fr_1, _p_to_1, _q_to_1)
    end
end

function get_batch(olc, BS)
    if BS == olc.tm.shape
        return tm, ta, g, b, p_fr1, q_fr1, p_to1, q_to1
    else
        return  olc.tm.narrow(0, 0, BS),
                olc.ta.narrow(0, 0, BS),
                olc.g.narrow(0, 0, BS),
                olc.b.narrow(0, 0, BS),
                olc.p_fr1.narrow(0, 0, BS),
                olc.q_fr1.narrow(0, 0, BS),
                olc.p_to1.narrow(0, 0, BS),
                olc.q_to1.narrow(0, 0, BS)
    end

end

function get_Flows(olc::OhmsLawConstraint, vm, va)
    vm_fr = torch.index_select(vm, 1, olc.idx_bus_fr)
    vm_to = torch.index_select(vm, 1, olc.idx_bus_to)
    va_fr = torch.index_select(va, 1, olc.idx_bus_fr)
    va_to = torch.index_select(va, 1, olc.idx_bus_to)

    curr_BS = vm.shape[1]
    tm, ta, g, b, p_fr1, q_fr1, p_to1, q_to1 = get_batch(olc, curr_BS)

    # element wise ops on tensor
    f2 = (vm_fr / tm)^2
    f3 = vm_fr / tm * vm_to
    _f3 = (va_fr - va_to - ta)
    f3c, f3s = torch.cos(_f3), torch.sin(_f3)
    p_fr = (p_fr1 * f2) - (g * f3 * f3c) - (b * f3 * f3s)
    q_fr = (q_to1 * f2) + (b * f3 * f3c) - (g * f3 * f3s)

    t2 = vm_to^2
    t3 = vm_to * vm_fr / tm
    _t3 = (va_to - va_fr + ta)
    t3c, t3s = torch.cos(_t3), torch.sin(_t3)
    p_to = (p_to1 * t2) - (g * t3 * t3c) - (b * t3 * t3s)
    q_to = (q_to1 * t2) + (b * t3 * t3c) - (g * t3 * t3s)

    return torch.cat((p_fr, q_fr, p_to, q_to), 1)
end

##########
