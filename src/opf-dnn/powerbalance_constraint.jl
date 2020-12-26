using JSON, JuMP, Ipopt, PowerModels
PowerModels.silence() # suppress warning and info messages
include("torch.jl")

"""
    Struct to encode the power balance constraint.
    Given the (predicted) active `pg` and reactive `qg` dispatch,
          the original active `pd` and reactive `qd` loads,
          the predicted power voltage magnitude `vm`,
          and the predicted power flows `PF`, `PT`
          check if :

              sum(pg) - sum(pd) - sum(gs) * vm^2 == PF
              sum(qg) - sum(qd) + sum(bs) * vm^2 == PT
"""
mutable struct PowerBalanceConstraint
    device::Any
    nbuses::Any
    bus_pg_idx::Any
    bus_mask_pg::Any
    bus_pd_idx::Any
    bus_mask_pd::Any
    bus_vm_idx::Any
    bus_gs_vals::Any
    f_buses_idx::Any
    t_buses_idx::Any

    function PowerBalanceConstraint(datainfo, config, device=nothing)
        if device == nothing
            device = torch.device("cpu")
        end
        BS = config["batchsize"]
        pnet = datainfo.powernet
        pm = PowerModels.build_model(pnet, ACPPowerModel, PowerModels.post_opf)
        N = datainfo.n_buses
        bus_IDs = [parse(Int, i) for i in keys(pm.data["bus"])]

        bus_pg_idx = [get_Pow_bus_idx(pm, bus_i, datainfo, "pg") for bus_i in bus_IDs]
        mask_pg    = [i[1] == -1 ? 0.0 : 1.0 for i in bus_pg_idx]
        bus_pg_idx = [i[1] == -1 ? iTen([0]).to(device) : iTen(i.-1).to(device) for i in bus_pg_idx]
        mask_pg = Ten(repeat([mask_pg]; outer=BS)).to(device)

        bus_pd_idx = [get_Pow_bus_idx(pm, bus_i, datainfo, "pd") for bus_i in bus_IDs]
        mask_pd    = [i[1] == -1 ? 0.0 : 1.0 for i in bus_pd_idx]
        bus_pd_idx = [i[1] == -1 ? iTen([0]).to(device) : iTen(i.-1).to(device) for i in bus_pd_idx]
        mask_pd = Ten(repeat([mask_pd]; outer=BS)).to(device)

        bus_gs = [get_bus_shunt(pm, bus_i, "gs") for bus_i in bus_IDs]
        bus_gs = Ten(repeat([bus_gs]; outer=BS)).to(device)

        bus_vm_idx = [get_Vol_bus_idx(pm, bus_i, datainfo, "vm") for bus_i in bus_IDs]
        bus_vm_idx = iTen(bus_vm_idx.-1).to(device) ## only 1

        f_buses = [[k for (k,v) in pnet["branch"] if v["f_bus"] == bus_i] for bus_i in bus_IDs]
        t_buses = [[k for (k,v) in pnet["branch"] if v["t_bus"] == bus_i] for bus_i in bus_IDs]
        f_buses_idx = [iTen([get_keypos(datainfo, "pf", i) for i in F].-1).to(device) for F in f_buses]
        t_buses_idx = [iTen([get_keypos(datainfo, "pt", i) for i in T].-1).to(device) for T in t_buses]

        new(device, N,
            bus_pg_idx, mask_pg,
            bus_pd_idx, mask_pd,
            bus_vm_idx, bus_gs,
            f_buses_idx, t_buses_idx)
    end

    function get_bus_shunt(pm, bus_idx, type, nw=0, cnd=1)
        bus_shunts = ref(pm, nw, :bus_shunts, bus_idx)
        vals = sum0([ref(pm, nw, :shunt, j, type, cnd) for j in bus_shunts])
    end

    function get_Vol_bus_idx(pm, bus_idx, datainfo, pred_type)
        @assert pred_type in ["vm", "va"]
        vals = get_keypos(datainfo, pred_type, string(bus_idx))
        return vals == nothing ? -1 : vals
    end

    function get_Pow_bus_idx(pm, bus_idx, datainfo, pred_type, nw=0, cnd=1)
        @assert pred_type in ["pg", "qg", "pd", "qd"]
        pm_type = pred_type in ["pg", "qg"] ? :bus_gens : :bus_loads
        # Take all buses related to index `i`
        bus_elem = ref(pm, nw, pm_type, bus_idx)
        if length(bus_elem) == 0
            vals = [-1]
        else
            # the indexes of the vector `pd` (or its prediction) corresponding to bus_elem
            vals = [get_keypos(datainfo, pred_type, string(b)) for b in bus_elem]
        end
        return vals[1] == nothing ? [-1] : vals
    end
end

function get_losses(pbc::PowerBalanceConstraint, pd, pg, vm, pf, pt)
    N = pbc.nbuses
    curr_BS = pg.shape[1]

    if curr_BS != pbc.bus_mask_pg.shape
        mask_pg = pbc.bus_mask_pg.narrow(0, 0, curr_BS)
        mask_pd = pbc.bus_mask_pd.narrow(0, 0, curr_BS)
        Gs = pbc.bus_gs_vals.narrow(0, 0, curr_BS)
    else
        mask_pg = pbc.bus_mask_pg
        mask_pd = pbc.bus_mask_pd
        Gs = pbc.bus_gs_vals
    end

    Pg = torch.stack([torch.sum(torch.index_select(pg, 1, pbc.bus_pg_idx[i]), dim=1) for i in 1:N]).t() * mask_pg
    Pd = torch.stack([torch.sum(torch.index_select(pd, 1, pbc.bus_pd_idx[i]), dim=1) for i in 1:N]).t() * mask_pd
    Vm = torch.index_select(vm, 1, pbc.bus_vm_idx)
    Ff = torch.stack([torch.sum(torch.index_select(pf, 1, pbc.f_buses_idx[i]), dim=1) for i in 1:N]).t()
    Ft = torch.stack([torch.sum(torch.index_select(pt, 1, pbc.t_buses_idx[i]), dim=1) for i in 1:N]).t()

    # pf_loss = (Ff + Ft) - (Pg - Pd - (Gs * Vm^2))
    # return torch.mean(pf_loss)
    return F.l1_loss((Ff + Ft), (Pg - Pd - (Gs * Vm^2)))
end


##################################
