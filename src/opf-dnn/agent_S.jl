using Printf, ProgressMeter, Statistics
include("torch.jl")
include("networks.jl")
include("utils.jl")
include("datautils.jl")
include("constraints.jl")

""" OPF Agent class """
mutable struct OpfAgent
    config::Dict
    device::PyObject
    model::Any
    optimizer::Any
    loss::Dict

    train_loader::DataLoader
    test_loader::DataLoader
    constraints::Constraints
    datainfo::DataInfo
    verbose::Bool
    data_indexes::Any

    function OpfAgent(args)
        # Select Device
        device = torch.device(ifelse(!args["nocuda"] && torch.cuda.is_available(), "cuda", "cpu"))
        # Read Data and construct auxiliary structures
        data = read_data(args)
        datainfo = DataInfo(data, args)
        constr   = Constraints(data, datainfo, args, device)

        # Load Datasets
        # (Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out, data_indexes) = load_setpoint_datasets(data, datainfo, args)
        (Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out, data_indexes) = load_setpoint_datasets_Perc(data, datainfo, args)

        train, test = DataLoader(Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out,
                                 shuffle=false, batchsize=args["batchsize"])

        indims = Dict("Sd" => size(Sd)[2], "vm" => size(vm_in)[2],
                      "va" => size(va_in)[2], "pg" => size(pg_in)[2])
        outdims = Dict("vm" => size(vm_out)[2], "va" => size(va_out)[2],
                       "pg" => size(pg_out)[2])

        # Create Model and Set optimizers
        model = DeepFFNetPV(indims, outdims, args["nettype"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args["lr"], betas=(0.9, 0.999))

        loss = Dict("vm" => F.mse_loss, "va" => F.mse_loss,
                    "pg" => F.mse_loss, "ohm" => F.mse_loss,
                    "vm-bnd" => partial(bound_penalty, constr.bnd["vm-bnd"]),
                    "va-bnd" => partial(bound_penalty, constr.bnd["va-bnd"]),
                    "pg-bnd" => partial(bound_penalty, constr.bnd["pg-bnd"]),
                    "flow-bnd" => partial(bound_penalty, constr.bnd["flow-bnd"])
                    )

        new(args, device, model, optimizer, loss,
            train, test, constr, datainfo, args["verbose"],
            data_indexes)
    end
end # module

# todo continue from here
function train(agent::OpfAgent)
    """
    Main training loop
    """
    nepochs = agent.config["nepochs"]
    device = agent.device
    batchsize = agent.config["batchsize"]

    _lkeys = ["vm", "va", "vm-bnd", "va-bnd", "pg", "pg-bnd",
              "ohm", "flow-bnd", "klc"]
    _losses = Dict(k => [] for k in _lkeys)
    L = Dict{String,Any}(k => nothing for k in _lkeys)
    con = agent.constraints

    #update every 10 seconds
    barlen=round(Int, nepochs*agent.train_loader.n / batchsize)
    p = Progress(barlen, dt=0.1, color=:blue)
    for epoch in 1:nepochs
        let Lepoch = Dict{String,Any}(k => [] for k in _lkeys)
            for (i, (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg)) in enumerate(agent.train_loader)
                (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg) = Ten(sd).to(device),
                    Ten(ivm).to(device), Ten(iva).to(device), Ten(ipg).to(device),
                    Ten(oflow).to(device), Ten(Δ_vm).to(device), Ten(Δ_va).to(device),
                    Ten(Δ_pg).to(device)

                """ Predict Voltage Magnitudes and Angle"""
                Δ_vm_pred, Δ_va_pred, Δ_pg_pred = agent.model(sd, ivm, iva, ipg)
                vm_pred, va_pred, pg_pred = ivm + Δ_vm_pred, iva + Δ_va_pred, ipg + Δ_pg_pred
                #vm_pred, va_pred, pg_pred = vm_pred.to(device), va_pred.to(device), pg_pred.to(device)

                """ Predict FLows from va and vm """
                oflow_pred = get_Flows(agent.constraints.olc, vm_pred, va_pred)
                oSf, oSt = get_complex_flows(oflow_pred)

                """ Losses for vm """
                L["vm"] = agent.loss["vm"](Δ_vm_pred, Δ_vm)
                L["vm-bnd"]  = agent.loss["vm-bnd"](vm_pred)
                push!(_losses["vm"], L["vm"].item())
                push!(_losses["vm-bnd"], L["vm-bnd"].item())
                push!(Lepoch["vm-bnd"], L["vm-bnd"].item())

                """ Losses for va """
                L["va"] = agent.loss["va"](Δ_va_pred, Δ_va)
                L["va-bnd"]  = agent.loss["va-bnd"](va_pred)
                push!(_losses["va"], L["va"].item())
                push!(_losses["va-bnd"], L["va-bnd"].item())
                push!(Lepoch["va-bnd"], L["va-bnd"].item())

                """ Losses for pg """
                L["pg"] = agent.loss["pg"](Δ_pg_pred, Δ_pg)
                L["pg-bnd"]  = agent.loss["pg-bnd"](pg_pred)
                push!(_losses["pg"], L["pg"].item())
                push!(_losses["pg-bnd"], L["pg-bnd"].item())
                push!(Lepoch["pg-bnd"], L["pg-bnd"].item())

                """ Losses for Flows and RateA Losses """
                L["flow-bnd"] = agent.loss["flow-bnd"](oSf) + agent.loss["flow-bnd"](oSt)
                push!(_losses["flow-bnd"], L["flow-bnd"].item())
                push!(Lepoch["flow-bnd"], L["flow-bnd"].item())

                L["ohm"] = agent.loss["ohm"](oflow_pred, oflow)
                push!(_losses["ohm"], L["ohm"].item())
                push!(Lepoch["ohm"], L["ohm"].item())

                """ Power blanace constraint losses """
                (_, pd_o, _, _) = get_components_from_Sd(sd, agent.datainfo)
                (pf_pred, _, pt_pred, _) = get_components_from_Sij(oflow_pred, agent.datainfo)
                L["klc"] = get_losses(agent.constraints.pbc, pd_o, pg_pred, vm_pred, pf_pred, pt_pred)
                push!(_losses["klc"], L["klc"].item())
                push!(Lepoch["klc"], L["klc"].item())

                loss = get_losses(L, con, agent.config)
                backprop(agent, loss)

                !agent.verbose && next!(p)
                if agent.verbose && i % 100 == 0
                    println("Epoch: $epoch\t($i) ")
                    println("$(@sprintf("\tLosses: \tvm: %.6f \tbnd: %.6f", L["vm"].item(), L["vm-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tva: %.6f \tbnd: %.6f", L["va"].item(), L["va-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tpg: %.6f \tbnd: %.6f", L["pg"].item(), L["pg-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tflow: %.6f \tbnd: %.6f", L["ohm"].item(), L["flow-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tPBC: %.6f", L["klc"].item()))")
                end
            end
            # Lagrangian Step
            update_allλ!(con, Lepoch)
            @show [con.λ[ctype] for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]]
            GC.gc(false)
        end
    end
    return _losses
end

function get_losses(L, con, config)
    _losses = L["vm"] + L["va"] + L["pg"]
    if config["use-constraints"]
        if config["use-dual-update"]
            for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]
                _losses += L[ctype]
            end
        else
            for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]
                _losses += con.λ[ctype] * L[ctype]
            end
        end
    end
    return _losses
end

function test(agent::OpfAgent)
    results = Dict()
    N = agent.test_loader.n
    err = Dict("vm" => [], "va" => [], "pg" => [],
               "ohm" => [],  # error-pred
               "klc" => [])

   batchsize = agent.config["batchsize"]
   barlen=round(Int, agent.test_loader.n / batchsize)
   p = Progress(barlen, dt=1, color=:blue)
    let device = agent.device
        @pywith torch.no_grad() begin
        for (i, (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg)) in enumerate(agent.test_loader)
            # Update iterators in auxilary structures
            #update_iterators(agent, "train")
            (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg) = Ten(sd).to(device),
                Ten(ivm).to(device), Ten(iva).to(device), Ten(ipg).to(device),
                Ten(oflow).to(device), Ten(Δ_vm).to(device), Ten(Δ_va).to(device),
                Ten(Δ_pg).to(device)

                Δ_vm_pred, Δ_va_pred, Δ_pg_pred = agent.model(sd, ivm, iva, ipg)
                vm_pred, va_pred, pg_pred = ivm + Δ_vm_pred, iva + Δ_va_pred, ipg + Δ_pg_pred

                vm_pred = bclamp(agent.constraints.bnd["vm-bnd"], vm_pred)
                va_pred = bclamp(agent.constraints.bnd["va-bnd"], va_pred)
                pg_pred = bclamp(agent.constraints.bnd["pg-bnd"], pg_pred)

                ovm, ova, opg = ivm + Δ_vm, iva + Δ_va, ipg + Δ_pg
                oflow_pred = get_Flows(agent.constraints.olc, vm_pred, va_pred)

                (_, pd_o, _, _) = get_components_from_Sd(sd, agent.datainfo)
                (pf_pred, _, pt_pred, _) = get_components_from_Sij(oflow_pred, agent.datainfo)
                klc_loss = get_losses(agent.constraints.pbc, pd_o, pg_pred, vm_pred, pf_pred, pt_pred)

                push!(err["vm"], F.l1_loss(vm_pred, ovm).item())
                push!(err["va"], F.l1_loss(va_pred, ova).item())
                push!(err["pg"], F.l1_loss(pg_pred, opg).item())
                push!(err["ohm"], F.l1_loss(oflow_pred, oflow).item())
                push!(err["klc"], klc_loss.item())

                # Save output
                merge!(results, collect_results(sd, ovm, ova, opg, oflow,
                                               va_pred, vm_pred, pg_pred, oflow_pred,
                                               agent.datainfo))
               next!(p)
            end
            GC.gc(false)
            println("$(@sprintf("Test Errors: \tvm: %.6f \tva: %.6f \tpg: %.6f \tflow: %.6f \tflow-balance: %.6f",
                                mean(err["vm"]), mean(err["va"]), mean(err["pg"]), mean(err["ohm"]), mean(err["klc"]) ))")
        end
    end
    return results, err
end

function collect_results(sd, vm, va, pg, flow, va_pred, vm_pred, pg_pred, flow_pred, dinfo)
    #pd_true, qd_true = get_Sd_from_pred(sd, dinfo, true)
    pd_old, pd_true, qd_old, qd_true = get_components_from_Sd(sd, dinfo, true)

    vm_pred, va_pred, pg_pred = toarray(vm_pred), toarray(va_pred), toarray(pg_pred)
    vm_true, va_true, pg_true = toarray(vm), toarray(va), toarray(pg)
    pf_pred, qf_pred, pt_pred, qt_pred = get_flows(flow_pred, true)
    pf_true, qf_true, pt_true, qt_true = get_flows(flow, true)

    batchsize = min(size(sd)[1])
    results = Dict()
    for i in 1:batchsize
        test_index = agent.test_loader.current_indices[i]
        results[test_index] = Dict(
            "pd" => Dict(dinfo.keys["pd"][k] => v  for (k,v) in enumerate(pd_true[i,:])),
            "qd" => Dict(dinfo.keys["qd"][k] => v  for (k,v) in enumerate(qd_true[i,:])),
            "vm" => Dict(dinfo.keys["vm"][k] => v  for (k,v) in enumerate(vm_true[i,:])),
            "va" => Dict(dinfo.keys["va"][k] => v  for (k,v) in enumerate(va_true[i,:])),
            "pg" => Dict(dinfo.keys["pg"][k] => v  for (k,v) in enumerate(pg_true[i,:])),
            "pf" => Dict(dinfo.keys["pf"][k] => v  for (k,v) in enumerate(pf_true[i,:])),
            "qf" => Dict(dinfo.keys["qf"][k] => v  for (k,v) in enumerate(qf_true[i,:])),
            "pt" => Dict(dinfo.keys["pt"][k] => v  for (k,v) in enumerate(pt_true[i,:])),
            "qt" => Dict(dinfo.keys["qt"][k] => v  for (k,v) in enumerate(qt_true[i,:])),
            "pred-vm" => Dict(dinfo.keys["vm"][k] => v  for (k,v) in enumerate(vm_pred[i,:])),
            "pred-va" => Dict(dinfo.keys["va"][k] => v  for (k,v) in enumerate(va_pred[i,:])),
            "pred-pg" => Dict(dinfo.keys["pg"][k] => v  for (k,v) in enumerate(pg_pred[i,:])),
            "pred-pf" => Dict(dinfo.keys["pf"][k] => v  for (k,v) in enumerate(pf_pred[i,:])),
            "pred-qf" => Dict(dinfo.keys["qf"][k] => v  for (k,v) in enumerate(qf_pred[i,:])),
            "pred-pt" => Dict(dinfo.keys["pt"][k] => v  for (k,v) in enumerate(pt_pred[i,:])),
            "pred-qt" => Dict(dinfo.keys["qt"][k] => v  for (k,v) in enumerate(qt_pred[i,:]))
            )
    end
    return results
end
