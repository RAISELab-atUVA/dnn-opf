using JSON
using PowerModels
PowerModels.silence() # suppress warning and info messages
include("dataloader.jl")
include("pypickle.jl")


""" Data Util Info Structure """
struct DataInfo
    keys::Any
    n_loads::Int
    n_gens::Int
    n_lines::Int
    n_buses::Int
    powernet::Any

    function DataInfo(data, args)
        pd_keys = collect(keys(data["experiments"][1]["pd"]))
        qd_keys = collect(keys(data["experiments"][1]["qd"]))
        pg_keys = collect(keys(data["experiments"][1]["pg"]))
        qg_keys = collect(keys(data["experiments"][1]["qg"]))
        vg_keys = collect(keys(data["experiments"][1]["vg"]))
        pf_keys = collect(keys(data["experiments"][1]["pf"]))
        qf_keys = collect(keys(data["experiments"][1]["qf"]))
        pt_keys = collect(keys(data["experiments"][1]["pt"]))
        qt_keys = collect(keys(data["experiments"][1]["qt"]))
        va_keys = collect(keys(data["experiments"][1]["va"]))
        vm_keys = collect(keys(data["experiments"][1]["vm"]))

        n_loads = length(data["experiments"][1]["pd"])
        n_gens  = length(data["experiments"][1]["pg"])
        n_lines = length(data["experiments"][1]["pf"])
        n_buses = length(data["experiments"][1]["va"])

        dkeys = Dict("pd" => pd_keys, "qd" => qd_keys,
                    "pg" => pg_keys, "qg" => qg_keys,
                    "vg" => vg_keys,
                    "pf" => pf_keys, "qf" => qf_keys,
                    "pt" => pt_keys, "qt" => qt_keys,
                    "va" => va_keys, "vm" => vm_keys)

        input_net = PowerModels.parse_file(args["path"] * args["netpath"] * args["netname"] * ".m")
        new(dkeys, n_loads, n_gens, n_lines, n_buses, input_net)
    end
end

""" Read the input Network File """
function read_data(config, filename=nothing, prefix="")
    println(pwd())
    path_data = config["path"] * "data/traindata/" * config["netname"] * "/"
    if filename == nothing
        file =  path_data * prefix * config["traindata"]
    else
        file =  path_data * prefix * filename
    end

    println("Reading: $(file)")
    if extension(file) == ".pkl"
        return read_pickle(file)
    else
        return JSON.parsefile(file)
    end
end

function write_data(data, config, prefix=nothing)
    file = get_file_name(config, "results", prefix)

    println("Writing: $(file)")
    if extension(file) == ".pkl"
        write_pickle(file, data)
    else
        open(file, "w") do f
            write(f, JSON.json(data, 4))
        end
    end
end

""" Create cartesian product of two 2-dimensional vectors """
function combo(x,y)
    leny=size(y)[1]
    lenx=size(x)[1]
    OUT = []
    for i = 1:lenx
        for j = 1:leny
            if i != j
                push!(OUT, vcat(x[i,:], y[j,:]))
            end
        end
    end
    return OUT
end

""" Get the indexes of the arrays associated to neighbors' scaling factors """
function get_neighbors_idx(data, range)
    scale = [exp["scale"] for exp in data["experiments"]]
    lb, ub = minimum(scale), maximum(scale)
    vScaleIdx = []
    let _lb = lb
        while _lb < ub
            _ub = min(_lb+range, ub)
            push!(vScaleIdx, [i for (i, v) in enumerate(scale) if _lb <= v < _ub])
            _lb = _ub
        end
    end
    return vScaleIdx
end

""" Return the index whose of corresponding to `keyname`
    in the keys[prop] vector of datainfo
"""
function get_keypos(di::DataInfo, prop, keyname)
    L = findall(x->x == keyname, di.keys[prop])
    return length(L) > 0 ? L[1] : nothing
end

""" Reads Input Data """
function load_naive_datasets(data, datainfo, config)
    # inputs
    pd = hcat([collect(values(exp["pd"])) for exp in data["experiments"]]...)'
    qd = hcat([collect(values(exp["qd"])) for exp in data["experiments"]]...)'
    Sd = hcat(pd, qd)

    # outputs
    pg = hcat([collect(values(exp["pg"])) for exp in data["experiments"]]...)'
    qg = hcat([collect(values(exp["qg"])) for exp in data["experiments"]]...)'
    vm = hcat([collect(values(exp["vm"])) for exp in data["experiments"]]...)'
    va = hcat([collect(values(exp["va"])) for exp in data["experiments"]]...)'
    pf = hcat([collect(values(exp["pf"])) for exp in data["experiments"]]...)'
    qf = hcat([collect(values(exp["qf"])) for exp in data["experiments"]]...)'
    pt = hcat([collect(values(exp["pt"])) for exp in data["experiments"]]...)'
    qt = hcat([collect(values(exp["qt"])) for exp in data["experiments"]]...)'

    Flows = hcat(pf, qf, pt, qt)
    n = length(data["experiments"])
    data_indexes = hcat(1:n, 1:n)

    return (Sd, Flows, vm, va, pg, qg, data_indexes, pd) #modified
    #return (Sd, Flows, vm, va, pg, qg, data_indexes)
end

function load_setpoint_datasets(data, datainfo, config)
    # Construct the data loaders
    vScaleIdx = get_neighbors_idx(data, config["range"])
    pd = hcat([collect(values(exp["pd"])) for exp in data["experiments"]]...)'
    qd = hcat([collect(values(exp["qd"])) for exp in data["experiments"]]...)'
    combo_pd = hcat(vcat([combo(pd[S,:], pd[S,:]) for S in vScaleIdx]...)...)'#Killed
    combo_qd = hcat(vcat([combo(qd[S,:], qd[S,:]) for S in vScaleIdx]...)...)'

    pf = hcat([collect(values(exp["pf"])) for exp in data["experiments"]]...)'
    qf = hcat([collect(values(exp["qf"])) for exp in data["experiments"]]...)'
    pt = hcat([collect(values(exp["pt"])) for exp in data["experiments"]]...)'
    qt = hcat([collect(values(exp["qt"])) for exp in data["experiments"]]...)'
    combo_pf = hcat(vcat([combo(pf[S,:], pf[S,:]) for S in vScaleIdx]...)...)'
    combo_qf = hcat(vcat([combo(qf[S,:], qf[S,:]) for S in vScaleIdx]...)...)'
    combo_pt = hcat(vcat([combo(pt[S,:], pt[S,:]) for S in vScaleIdx]...)...)'
    combo_qt = hcat(vcat([combo(qt[S,:], qt[S,:]) for S in vScaleIdx]...)...)'

    vm = hcat([collect(values(exp["vm"])) for exp in data["experiments"]]...)'
    va = hcat([collect(values(exp["va"])) for exp in data["experiments"]]...)'
    combo_vm = hcat(vcat([combo(vm[S,:], vm[S,:]) for S in vScaleIdx]...)...)'
    combo_va = hcat(vcat([combo(va[S,:], va[S,:]) for S in vScaleIdx]...)...)'

    pg = hcat([collect(values(exp["pg"])) for exp in data["experiments"]]...)'
    combo_pg = hcat(vcat([combo(pg[S,:], pg[S,:]) for S in vScaleIdx]...)...)'

    data_indexes = hcat(vcat([combo(S, S) for S in vScaleIdx]...)...)'
    
    nbuses = datainfo.n_buses
    combo_vm_in = combo_vm[:,1:nbuses]
    combo_vm_out = combo_vm[:,nbuses+1:2*nbuses]
    Δ_vm_out = combo_vm_out .- combo_vm_in
    combo_va_in = combo_va[:,1:nbuses]
    combo_va_out = combo_va[:,nbuses+1:2*nbuses]
    Δ_va_out = combo_va_out .- combo_va_in

    nlines = datainfo.n_lines
    combo_pf_out = combo_pf[:, nlines+1 : 2*nlines]
    combo_qf_out = combo_qf[:, nlines+1 : 2*nlines]
    combo_pt_out = combo_pt[:, nlines+1 : 2*nlines]
    combo_qt_out = combo_qt[:, nlines+1 : 2*nlines]

    ngens = datainfo.n_gens
    combo_pg_in = combo_pg[:,1:ngens]
    combo_pg_out = combo_pg[:,ngens+1:2*ngens]
    Δ_pg_out = combo_pg_out .- combo_pg_in

    (Sd, vm_in, va_in, pg_in) = hcat(combo_pd, combo_qd),
                                hcat(combo_vm_in),
                                hcat(combo_va_in),
                                hcat(combo_pg_in)
    Flows_out = hcat(combo_pf_out, combo_qf_out, combo_pt_out, combo_qt_out)
    vm_out = hcat(Δ_vm_out)
    va_out = hcat(Δ_va_out)
    pg_out = hcat(Δ_pg_out)

    return (Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out,
            data_indexes)
end

function load_setpoint_datasets_Perc(data, datainfo, config)
    scale = [exp["scale"] for exp in data["experiments"]]
    lb, ub = minimum(scale), maximum(scale)
    dist = (args["state-distance"] / 100)
    combos = []
    for (is, curr_state) in enumerate(scale)
          next_states = [i for (i, v) in enumerate(scale) if curr_state <= v < curr_state + dist]
          combos = vcat(combos, [[is, ns] for ns in next_states])
          combos = vcat(combos, [[ns, is] for ns in next_states])
    end
    println(length(combos))
    N = min(config["traindata-size"], length(combos))
    combos = combos[sample(1:length(combos), N, replace=false)]

    pd = hcat([collect(values(exp["pd"])) for exp in data["experiments"]]...)'
    qd = hcat([collect(values(exp["qd"])) for exp in data["experiments"]]...)'
    combo_pd = hcat([vcat(pd[C[1],:], pd[C[2],:]) for C in combos]...)'
    combo_qd = hcat([vcat(qd[C[1],:], qd[C[2],:]) for C in combos]...)'

    pf = hcat([collect(values(exp["pf"])) for exp in data["experiments"]]...)'
    qf = hcat([collect(values(exp["qf"])) for exp in data["experiments"]]...)'
    pt = hcat([collect(values(exp["pt"])) for exp in data["experiments"]]...)'
    qt = hcat([collect(values(exp["qt"])) for exp in data["experiments"]]...)'
    combo_pf = hcat([vcat(pf[C[1],:], pf[C[2],:]) for C in combos]...)'
    combo_qf = hcat([vcat(qf[C[1],:], qf[C[2],:]) for C in combos]...)'
    combo_pt = hcat([vcat(pt[C[1],:], pt[C[2],:]) for C in combos]...)'
    combo_qt = hcat([vcat(qt[C[1],:], qt[C[2],:]) for C in combos]...)'

    vm = hcat([collect(values(exp["vm"])) for exp in data["experiments"]]...)'
    va = hcat([collect(values(exp["va"])) for exp in data["experiments"]]...)'
    combo_vm = hcat([vcat(vm[C[1],:], vm[C[2],:]) for C in combos]...)'
    combo_va = hcat([vcat(va[C[1],:], va[C[2],:]) for C in combos]...)'

    pg = hcat([collect(values(exp["pg"])) for exp in data["experiments"]]...)'
    combo_pg = hcat([vcat(pg[C[1],:], pg[C[2],:]) for C in combos]...)'

    data_indexes = hcat(combos[:,1]...)'

    println(data_indexes)

    nbuses = datainfo.n_buses
    combo_vm_in = combo_vm[:,1:nbuses]
    combo_vm_out = combo_vm[:,nbuses+1:2*nbuses]
    Δ_vm_out = combo_vm_out .- combo_vm_in

    combo_va_in = combo_va[:,1:nbuses]
    combo_va_out = combo_va[:,nbuses+1:2*nbuses]
    Δ_va_out = combo_va_out .- combo_va_in

    nlines = datainfo.n_lines
    combo_pf_out = combo_pf[:, nlines+1 : 2*nlines]
    combo_qf_out = combo_qf[:, nlines+1 : 2*nlines]
    combo_pt_out = combo_pt[:, nlines+1 : 2*nlines]
    combo_qt_out = combo_qt[:, nlines+1 : 2*nlines]

    ngens = datainfo.n_gens
    combo_pg_in = combo_pg[:,1:ngens]
    combo_pg_out = combo_pg[:,ngens+1:2*ngens]
    Δ_pg_out = combo_pg_out .- combo_pg_in


    (Sd, vm_in, va_in, pg_in) = hcat(combo_pd, combo_qd),
                              hcat(combo_vm_in),
                              hcat(combo_va_in),
                              hcat(combo_pg_in)
    Flows_out = hcat(combo_pf_out, combo_qf_out, combo_pt_out, combo_qt_out)
    vm_out = hcat(Δ_vm_out)
    va_out = hcat(Δ_va_out)
    pg_out = hcat(Δ_pg_out)

    return (Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out,
            data_indexes)
end



""" Reads Data Constraints """
# deprecated
function get_data_constraints(data, config)
    vglim = hcat(collect(values(data["constraints"]["vg_lim"]))...)'
    pglim = hcat(collect(values(data["constraints"]["pg_lim"]))...)'
    qglim = hcat(collect(values(data["constraints"]["qg_lim"]))...)'
    line_rx = hcat(collect(values(data["constraints"]["qg_lim"]))...)'
    vmlim = hcat(collect(values(data["constraints"]["vm_lim"]))...)'
    ratea = hcat(collect(values(data["constraints"]["rate_a"]))...)'

    vlen, plen = length(vglim), length(pglim)

    return vglim, pglim
end

""" Retuns (pd, qd) or (pd_in, pd_out, qd_in, qd_out) from tensor Sd """
function get_components_from_Sd(Sd, info::DataInfo, numpy=false)
    n = info.n_loads
    if Sd.shape[2] == 2*n
        pd = Sd.narrow(1, 0, n)
        qd = Sd.narrow(1, n, n)
        if numpy
            pd, qd = pd.cpu().numpy(), qd.cpu().numpy()
        end
        return (pd, qd)
    elseif Sd.shape[2] == 4*n
        pd_i = Sd.narrow(1, 0,   n)
        pd_o = Sd.narrow(1, 1*n, n)
        qd_i = Sd.narrow(1, 2*n, n)
        qd_o = Sd.narrow(1, 3*n, n)
        if numpy
            pd_i, pd_o, qd_i, qd_o = pd_i.cpu().numpy(), pd_o.cpu().numpy(),
                                     qd_i.cpu().numpy(), qd_o.cpu().numpy()
        end
        return (pd_i, pd_o, qd_i, qd_o)
    end
end

""" Return acrive and reactive flow p_ij, q_ij from complex Flow S_ij.
    If S_ij describes both flow from and flow to, then it returns,
    p_from, q_from, p_to, q_to
"""
function get_components_from_Sij(Sij, info::DataInfo, numpy=false)
    n = info.n_lines
    if Sij.shape[2] == 2*n
        pij = Sij.narrow(1, 0, n)
        qij = Sij.narrow(1, n, n)
        if numpy
            pij, qij = pij.cpu().numpy(), qij.cpu().numpy()
        end
        return (pij, qij)
    elseif Sij.shape[2] == 4*n
        pfr = Sij.narrow(1, 0,   n)
        qfr = Sij.narrow(1, 1*n, n)
        pto = Sij.narrow(1, 2*n, n)
        qto = Sij.narrow(1, 3*n, n)
        if numpy
            pfr, qfr, pto, qto = pfr.cpu().numpy(), qfr.cpu().numpy(),
                                  pto.cpu().numpy(), qto.cpu().numpy()
        end
        return (pfr, qfr, pto, qto)
    end
end


""" Retuns (pd, qd) from tensor
    Deprecated: use get_components_from_Sd instead """
function get_Sd_from_pred(x_tensor, info::DataInfo, numpy=false)
    pd = x_tensor.narrow(1, 0, info.n_loads)
    qd = x_tensor.narrow(1, info.n_loads, info.n_loads)
    if numpy
        return (pd.cpu().numpy(), qd.cpu().numpy())
    else
        return (pd, qd)
    end
end



""" Retuns (pg, qg) or (pg, vg) based on the version """
function get_Sg_from_pred(y_tensor, info::DataInfo, numpy=false)
    pg = y_tensor.narrow(1, 0, info.n_gens)
    qg = y_tensor.narrow(1, info.n_gens, info.n_gens)
    if numpy
        return (pg.cpu().numpy(), qg.cpu().numpy())
    else
        return (pg, qg)
    end
end

function get_Sl_from_pred(z_tensor, info::DataInfo, numpy=false)
    n = info.n_lines
    pf = z_tensor.narrow(1, 0, n)
    qf = z_tensor.narrow(1, n, n)
    pt = z_tensor.narrow(1, 2*n, n)
    qt = z_tensor.narrow(1, 3*n, n)
    if numpy
        return (pf.cpu().numpy(), qf.cpu().numpy(), pt.cpu().numpy(), qt.cpu().numpy())
    else
        return (pf, qf, pt, qt)
    end
end

""" Returns the complex flow from the active (p) and reactive (q) predictions
    tensor [p ... | q ...]
"""
function get_Sij_from_pq(pq_tensor, numpy=false)
    n = Int(pq_tensor.size()[2] / 2)
    p = pq_tensor.narrow(1, 0, n)
    q = pq_tensor.narrow(1, n, n)
    res = p^2 + q^2
    if numpy
        res =  res.detach().cpu().numpy()
    end
    return res
end

""" Compute complex flows from (pf, qf, pt, qt) """
function get_complex_flows(x, numpy=false)
    pf, qf, pt, qt = get_flows(x)
    Sf = pf^2 + qf^2
    St = pt^2 + qt^2
    if numpy
        Sf = toarray(Sf)
        St = toarray(Sf)
    end
    return Sf, St
end

""" Return the decomposed vector of flows given the flow tensor (pf, qf, pt, qt) """
function get_flows(x, numpy=false)
    n = Int(x.size()[2] / 4) # n lines
    pf = x.narrow(1, 0, n)
    qf = x.narrow(1, n, n)
    pt = x.narrow(1, 2*n, n)
    qt = x.narrow(1, 3*n, n)
    if numpy
        pf, qf, pt, qt = toarray(pf), toarray(qf), toarray(pt), toarray(qt)
    end
    return pf, qf, pt, qt
end

function get_V_from_pred(w_tensor, info::DataInfo, numpy=false)
    n = info.n_buses
    vm = w_tensor.narrow(1, 0, n)
    va = w_tensor.narrow(1, n, n)
    if numpy
        return (vm.cpu().numpy(), va.cpu().numpy())
    else
        return (vm, va)
    end
end

""" Return the index in the tensor given a name (e.g., the name of the generator) and feature (e.g., gen) """
function get_tensor_idx(name, feat, info::DataInfo)
    return findall(x->x==string(name), info.keys[feat])[1]
end

function save_model(agent, config, prefix=nothing)
    file = get_file_name(config, "model", prefix)
    nepochs = config["nepochs"]
    mkpath("data/predictionsC/" * config["netname"] * "/$nepochs/")
    mkpath("data/predictionsnoC/" * config["netname"] * "/$nepochs/")
    torch.save(agent.model.state_dict(), file)
end

function load_model!(agent, config, prefix=nothing)
    file = get_file_name(config, "model", prefix)
    println("Loading: $(file)")
    state_dict = torch.load(file, map_location=agent.device)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
end


# function load_checkpoint(file_name)
#     """
#     Latest checkpoint loader
#     :param file_name: name of the checkpoint file
#     """
#     #TODO
# end

# function save_checkpoint(file_name="checkpoint.pth.tar", is_best=0)
#     """
#     Checkpoint saver
#     :param file_name: name of the checkpoint file
#     :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
#     """
#    TODO
# end
