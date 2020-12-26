using JSON, JuMP, Ipopt, PowerModels
PowerModels.silence() # suppress warning and info messages
using ProgressMeter
include("opf_constraint_verifier.jl")

""" Take the argmax of a vector of pairs using either the first or the second
    dimensions """
function _argmax_pairs(v, dim=1)
    itr = Iterators.Stateful(v)
    a,b = dim == 1 ? (1,2) : (2,1)
    i_max, v_max = first(itr)[a], first(itr)[b]
    @inbounds for it in itr
        it[b] > v_max && (i_max = it[a]; v_max = it[b])
    end
    return i_max#, v_max
end

""" Create Model """
function get_mappings(data, ori_pg)
    pm = build_model(data, ACPPowerModel, PowerModels.post_opf,
                     setting = Dict("output" => Dict("branch_flows" => true)))

    gMdl_to_gSrc, gSrc_to_gMdl = Dict(), Dict()
    vMdl_to_vSrc, vSrc_to_vMdl = Dict(), Dict()
    for (i, gen) in pm.ref[:nw][pm.cnw][:gen]
        if string(gen["index"]) in keys(ori_pg)
            src_name = string(gen["index"])
        # if string(gen["source_id"][2]) in keys(ori_pg)
        #     src_name = string(gen["source_id"][2])
            gMdl_to_gSrc[i] = src_name
            gSrc_to_gMdl[src_name] = i

            bus_id = gen["gen_bus"]
            vMdl_to_vSrc[bus_id] = src_name
            vSrc_to_vMdl[src_name] = bus_id
        else
             gMdl_to_gSrc[i] = nothing
             bus_id = gen["gen_bus"]
             vMdl_to_vSrc[bus_id] = nothing
         end
     end
     return gMdl_to_gSrc, gSrc_to_gMdl, vMdl_to_vSrc, vSrc_to_vMdl
 end

""" Update Loads and Generators Values """
function update_net(data, pd, qd, pred_pg, pred_vg)
    newdata = deepcopy(data)
    for (k, ld) in newdata["load"]
        ld["pd"] = pd[k]
        ld["qd"] = qd[k]
    end

    for (k, val) in pred_pg
        newdata["gen"][k]["pg"] = val
    end

    for (k, val) in pred_vg
        newdata["gen"][k]["vg"] = val
    end
   return newdata
end

function relax_net(data, pd, qd, pred_pg, pred_vg)
    newdata = deepcopy(data)
    for (k, ld) in newdata["load"]
        ld["pd"] = pd[k]
        ld["qd"] = qd[k]
    end

    for (k, val) in pred_pg
        newdata["gen"][k]["pg"] = val
    end

    for (k, val) in pred_vg
        newdata["gen"][k]["vg"] = val
    end
    for (k, bus) in newdata["bus"]
        newdata["bus"][k]["vmax"] = 2.0
        newdata["bus"][k]["vin"] = 0.5
    end
    for (k, branch) in newdata["branch"]
        newdata["branch"][k]["rate_a"] = 9999
        newdata["branch"][k]["rata_b"] = 9999
        newdata["branch"][k]["rata_c"] = 9999
    end

   return newdata
end


""" Post OPF problem """
function find_restoration(data, solver)
    pm = PowerModels.build_model(data, ACPPowerModel, PowerModels.post_opf,
                     setting = Dict("output" => Dict("branch_flows" => true)))

    slack_gen = _argmax_pairs([(i, gen["pmax"] - gen["pg"]) for (i, gen) in pm.ref[:nw][pm.cnw][:gen]])
    slack_bus = collect(ids(pm, :ref_buses))[1]

    # Fix Voltage and Generators' power to prediction
    for (i, gen) in pm.ref[:nw][pm.cnw][:gen]
        if gen["pmax"] > 0 && i != slack_gen # gen["gen_bus"] != slack_bus
            bus_id = gen["gen_bus"]
            v_pg = PowerModels.var(pm, pm.cnw, pm.ccnd, :pg, i)
            @constraint(pm.model, v_pg >= gen["pg"])
            @constraint(pm.model, v_pg <= gen["pg"])

            v_vg = PowerModels.var(pm, pm.cnw, pm.ccnd, :vm, bus_id)
            @constraint(pm.model, v_vg >= gen["vg"])
            @constraint(pm.model, v_vg <= gen["vg"])
        end
    end
    gSrc = string(slack_gen)
    vSrc = string(pm.data["gen"][gSrc]["gen_bus"])
    gOri = pm.ref[:nw][pm.cnw][:gen][slack_gen]["pg"]
    vOri = pm.ref[:nw][pm.cnw][:gen][slack_gen]["vg"]

    sol = optimize_model!(pm, solver)

    # TODO - If solution exisits
    # compute slack difference
    slack_diff = Dict("pg" => abs(sol["solution"]["gen"][gSrc]["pg"] - gOri),
                      "vg" => abs(sol["solution"]["bus"][vSrc]["vm"] - vOri))
    return pm, sol, slack_diff
end

""" Find Feasible flow htat is the closest to current state """
function closest_feasible_dist(data, solver, pred_pg, pred_vg,
                               gSrc_to_gMdl, vSrc_to_vMdl, ac_sol, total_pg, total_qg)
   pm = PowerModels.build_model(data, ACPPowerModel, PowerModels.post_opf,
                     setting = Dict("output" => Dict("branch_flows" => true)))

    @objective(pm.model, Min,
        sum((PowerModels.var(pm, pm.cnw, pm.ccnd, :pg, gMdl) - pred_pg[gSrc])^2 for (gSrc, gMdl) in gSrc_to_gMdl)
        +
        sum((PowerModels.var(pm, pm.cnw, pm.ccnd, :vm, vMdl) - pred_vg[vSrc])^2 for (vSrc, vMdl) in vSrc_to_vMdl))

    sol = optimize_model!(pm, solver)
    pg_pred_err = [100.0*abs(sol["solution"]["gen"][string(gMdl)]["pg"] - pred_pg[gSrc])/total_pg for (gSrc, gMdl) in gSrc_to_gMdl]
    vm_pred_err = [100.0*abs(sol["solution"]["bus"][string(vMdl)]["vm"] - pred_vg[vSrc])/ac_sol["solution"]["bus"][string(vMdl)]["vm"] for (vSrc, vMdl) in vSrc_to_vMdl]
    pg_AC_err = [100.0*abs(sol["solution"]["gen"][string(gMdl)]["pg"] - ac_sol["solution"]["gen"][string(gMdl)]["pg"])/total_pg for (gSrc, gMdl) in gSrc_to_gMdl]
    vm_AC_err = [100.0*abs(sol["solution"]["bus"][string(vMdl)]["vm"] - ac_sol["solution"]["bus"][string(vMdl)]["vm"])/ac_sol["solution"]["bus"][string(vMdl)]["vm"] for (vSrc, vMdl) in vSrc_to_vMdl]

    pred_errors = Dict("max" => Dict("pg" => maximum(pg_pred_err), "vg" => maximum(vm_pred_err)),
                       "avg" => Dict("pg" => mean(pg_pred_err), "vg" => mean(vm_pred_err)),
                       "std" => Dict("pg" => std(pg_pred_err), "vg" => std(vm_pred_err)))
    AC_errors = Dict("max" => Dict("pg" => maximum(pg_AC_err), "vg" => maximum(vm_AC_err)),
                      "avg" => Dict("pg" => mean(pg_AC_err), "vg" => mean(vm_AC_err)),
                      "std" => Dict("pg" => std(pg_AC_err), "vg" => std(vm_AC_err)))

     n = pm.cnw
     opf_cost = 0
     for (i, gen) in pm.ref[:nw][n][:gen]
         gid = string(i)
         pg = sol["solution"]["gen"][gid]["pg"]

        if length(gen["cost"]) == 1
            opf_cost += gen["cost"][1]
        elseif length(gen["cost"]) == 2
            opf_cost += gen["cost"][1]*pg + gen["cost"][2]
        elseif length(gen["cost"]) == 3
            opf_cost += gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
        end
     end

    return sol, opf_cost, pred_errors, AC_errors
end

function check_opf_constraints(pm, sol)
    violations = Dict()

    v_gen, err = check_generator_bounds(pm, sol)
    violations["generator_bounds"] = v_gen
    v_vol, err = check_voltage_bounds(pm, sol)
    violations["voltage_bounds"] = v_vol

    #violations["angle_difference"] = check_angle_difference(pm, sol)
    violations["thermal_limits"] = check_thermal_limits(pm, sol)
    #violations["ohms_law"] = check_ohms_law(pm, sol)
    #violations["power_balance"] = check_power_balance(pm, sol)

    return violations
end

function solve_restoration_problem(settings, predicted_pv, comboidx)
    path_inputs = settings["netpath"]
    netname = settings["netname"]
    input_net = PowerModels.parse_file(path_inputs * netname * ".m")
    solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0, acceptable_tol=1e-6)

    train_data = Dict()
    let exp = read_data(settings, settings["traindata"])["experiments"]
        _keys = collect(keys(predicted_pv))
        train_data = Dict(k => exp[comboidx[k, 2]] for k in _keys)
    end

    _, gSrc_to_gMdl, _, vSrc_to_vMdl = get_mappings(input_net, first(values(predicted_pv))["pg"])

    #counters = Dict("fail"=>0, "succ"=>0, "feas"=>0)
    time = Dict("ori"=>[], "succ"=>[], "fail"=>[])
    #obj_diff = Dict("succ"=>[], "fail"=>[], "succ-slack"=>[])
    ML_violations = Dict()
    DC_violations = Dict()

    dc_pred_Pg_perc_errors = Dict()
    ac_pred_Pg_perc_errors = Dict()
    ac_pred_Vg_perc_errors = Dict()
    dc_ac_Pg_perc_errors   = Dict()

    ML_perc_error_pg = Dict()
    ML_perc_error_qg = Dict()
    ML_perc_error_vm = Dict()
    ML_deg_error_va = Dict()
    ML_error_flow = Dict()

    DC_OPF_Cost = Dict()
    AC_OPF_Cost = Dict()
    ML_Load_Flow_Cost = Dict()
    DC_Load_Flow_Cost = Dict()
    DC_OPF_Time = Dict()
    AC_OPF_Time = Dict()
    ML_Load_Flow_Time = Dict()
    DC_Load_Flow_Time = Dict()

    ## LoadFlow
    ML_loadflow_pred_perc_errors = Dict()
    ML_loadflow_acopf_perc_errors = Dict()
    DC_loadflow_dcopf_perc_errors = Dict()
    DC_loadflow_acopf_perc_errors = Dict()

    #########################
    # For all the predictions
    #########################
    n = length(predicted_pv)
    p = Progress(n, 0.1)   # minimum update interval: 1 second
    for (idx, data) in predicted_pv
        # Output values -- taken for consistency (Real, original, outputs)
        ori_pg, ori_vm, ori_va, ori_qg   = data["pg"], data["vm"], data["va"], train_data[idx]["qg"]
        ori_pf, ori_pt, ori_qf, ori_qt = data["pf"], data["pt"], data["qf"], data["qt"]
        ori_vg = Dict(k => ori_vm[string(vSrc_to_vMdl[k])] for k in keys(ori_pg))
        pd, qd           = data["pd"], data["qd"]
        pred_pg, pred_vm, pred_va = data["pred-pg"], data["pred-vm"], data["pred-va"]
        pred_pf, pred_pt, pred_qf, pred_qt = data["pred-pf"], data["pred-pt"], data["pred-qf"], data["pred-qt"]
        pred_vg = Dict(k => pred_vm[string(vSrc_to_vMdl[k])] for k in keys(ori_pg))
        total_pg = sum(ori_pg[k] for k in keys(ori_pg))
        total_qg = sum(ori_qg[k] for k in keys(ori_qg))

        ml_pg_err = [100.0*abs(ori_pg[k] - pred_pg[k])/total_pg  for k in keys(ori_pg)]
        ml_vm_err = [100.0*abs(ori_vm[k] - pred_vm[k])/ori_vm[k]  for k in keys(ori_vm)]
        ml_va_err = [(180.0/pi)*abs(ori_va[k] - pred_va[k])  for k in keys(ori_va)]
        ml_pf_err = [100.0*abs(ori_pf[k] - pred_pf[k])  for k in keys(ori_pf)]
        ml_pt_err = [100.0*abs(ori_pt[k] - pred_pt[k])  for k in keys(ori_pt)]
        ml_qf_err = [100.0*abs(ori_qf[k] - pred_qf[k])  for k in keys(ori_qf)]
        ml_qt_err = [100.0*abs(ori_qt[k] - pred_qt[k])  for k in keys(ori_qt)]
        avg_flow = (ml_pf_err .+ ml_pt_err .+ ml_qf_err .+ ml_qt_err) / 4.0

        ML_perc_error_pg[idx] = Dict("max" => maximum(ml_pg_err), "avg" => mean(ml_pg_err), "std" => std(ml_pg_err))
        ML_perc_error_vm[idx] = Dict("max" => maximum(ml_vm_err), "avg" => mean(ml_vm_err), "std" => std(ml_vm_err))
        ML_deg_error_va[idx] = Dict("max" => maximum(ml_va_err), "avg" => mean(ml_va_err), "std" => std(ml_va_err))
        ML_error_flow[idx] = Dict("max" => maximum(avg_flow), "avg" => mean(avg_flow), "std" => std(avg_flow))

        ori_time = train_data[idx]["solve_time"]
        ori_obj  = train_data[idx]["objective"]
        push!(time["ori"], ori_time)

        # Update network data
        test_net = update_net(input_net, pd, qd, pred_pg, pred_vg)

        dc_pm = PowerModels.build_model(test_net, DCPPowerModel, PowerModels.post_opf)
        dc_sol = optimize_model!(dc_pm, solver)
        ac_pm = PowerModels.build_model(test_net, ACPPowerModel, PowerModels.post_opf)
        ac_sol = optimize_model!(ac_pm, solver)
        AC_OPF_Cost[idx] = ac_sol["objective"]
        DC_OPF_Cost[idx] = dc_sol["objective"]
        AC_OPF_Time[idx] = ac_sol["solve_time"]
        DC_OPF_Time[idx] = dc_sol["solve_time"]

        if dc_sol["termination_status"] == LOCALLY_SOLVED && ac_sol["termination_status"] == LOCALLY_SOLVED
            # pg-DC - pg-ML
            dc_pred_errs = [100.0*abs(dc_sol["solution"]["gen"][string(gMdl)]["pg"] - pred_pg[gSrc])/total_pg for (gSrc, gMdl) in gSrc_to_gMdl]
            dc_pred_Pg_perc_errors[idx] =  Dict("max" => maximum(dc_pred_errs),
                                                "avg" => mean(dc_pred_errs),
                                                "std" => std(dc_pred_errs))

            # pg-AC - pg-ML
            ac_pred_errPg = [100.0*abs(ac_sol["solution"]["gen"][string(gMdl)]["pg"] - pred_pg[gSrc])/total_pg for (gSrc, gMdl) in gSrc_to_gMdl]
            ac_pred_errVg = [100.0*abs(ac_sol["solution"]["bus"][string(vMdl)]["vm"] - pred_vg[vSrc])/ac_sol["solution"]["bus"][string(vMdl)]["vm"] for (vSrc, vMdl) in vSrc_to_vMdl]

            ac_pred_Pg_perc_errors[idx] =  Dict("max" => maximum(ac_pred_errPg),
                                                "avg" => mean(ac_pred_errPg),
                                                "std" => std(ac_pred_errPg))
            ac_pred_Vg_perc_errors[idx] = Dict("max" => maximum(ac_pred_errVg),
                                                "avg" => mean(ac_pred_errVg),
                                                "std" => std(ac_pred_errVg))

            # pg-DC - pg-AC
            dc_ac_err = [100.0*abs(ac_sol["solution"]["gen"][string(gMdl)]["pg"] - dc_sol["solution"]["gen"][string(gMdl)]["pg"] )/total_pg for (gSrc, gMdl) in gSrc_to_gMdl]
            dc_ac_Pg_perc_errors[idx] = Dict("max" => maximum(dc_ac_err),
                                                "avg" => mean(dc_ac_err),
                                                "std" => std(dc_ac_err))
        end

        #############################
        # Load Flow (ML solution)
        #############################
        (ML_loadflow_sol, ML_loadflow_opf_cost,
        ML_loadflow_pred_perc_errors[idx],
        ML_loadflow_acopf_perc_errors[idx]) =
        closest_feasible_dist(test_net, solver, pred_pg, pred_vg, gSrc_to_gMdl, vSrc_to_vMdl, ac_sol,total_pg, total_qg)

        #############################
        # Load Flow (DC solution)
        #############################
        dc_vg = Dict(vSrc => 1.0 for (vSrc, vMdl) in vSrc_to_vMdl)
        dc_pg = Dict(gSrc => dc_sol["solution"]["gen"][string(gMdl)]["pg"] for (gSrc, gMdl) in gSrc_to_gMdl)
        (DC_loadflow_sol, DC_loadflow_opf_cost,
        DC_loadflow_dcopf_perc_errors[idx],
        DC_loadflow_acopf_perc_errors[idx]) =
        closest_feasible_dist(test_net, solver, dc_pg, dc_vg, gSrc_to_gMdl, vSrc_to_vMdl, ac_sol, total_pg, total_qg)

        #############################
        # OPF cost and Solve Time
        #############################
        ML_Load_Flow_Cost[idx] = ML_loadflow_opf_cost
        ML_Load_Flow_Time[idx] = ML_loadflow_sol["solve_time"]
        DC_Load_Flow_Cost[idx] = DC_loadflow_opf_cost
        DC_Load_Flow_Time[idx] = DC_loadflow_sol["solve_time"]

        #############################
        # Check constraint violations (ML)
        #############################
        ML_relaxed_net = relax_net(input_net, pd, qd, pred_pg, pred_vg)
        (ML_relaxloadflow_sol, _, _, _,) =
        closest_feasible_dist(ML_relaxed_net, solver, pred_pg, pred_vg, gSrc_to_gMdl, vSrc_to_vMdl, ac_sol, total_pg, total_qg)
        ML_violations[idx] = check_opf_constraints(ac_pm, ML_relaxloadflow_sol)

        # Check constraint violations (DC)
        DC_relaxed_net = relax_net(input_net, pd, qd, dc_pg, dc_vg)
        (DC_relaxloadflow_sol,  _, _, _) =
        closest_feasible_dist(DC_relaxed_net, solver, dc_pg, dc_vg, gSrc_to_gMdl, vSrc_to_vMdl, ac_sol, total_pg, total_qg)
        DC_violations[idx] = check_opf_constraints(ac_pm, DC_relaxloadflow_sol)

        next!(p)
    end

    return Dict(
                # Total number of items if all are sampled
                "total" => n,
                # Average time
                "avg_solve_time_ori" => mean(time["ori"]),

                ##################################
                # ML ERRORS (vs original data)
                ##################################
                # Active dispatch in max and average percentage (over total dispatch of all gen)
                "ML_perc_error_pg" => ML_perc_error_pg,
                # Reactive dispatch in max and average percentage (over total dispatch of all gen)
                #"ML_max_perc_error_qg" => ML_max_perc_error_qg,
                #"ML_avg_perc_error_qg" => ML_avg_perc_error_qg,
                # Voltage mag. percentage error (max and average) (over individual bus)
                "ML_perc_error_vm" => ML_perc_error_vm,
                # Voltage max/average angle error (absolute, but converted to degree from radian)
                "ML_deg_error_va" => ML_deg_error_va,
                # Power flow error (average over all the lines, converted to MW)
                # Power flow error, averaging over avg error of pf, pt, qf, and qt in MW
                "ML_error_flow" => ML_error_flow,

                ################################
                # PREDICTION ERRORS vs. AC and DC
                ################################
                # AC and prediction generator percentage distance
                "ac_pred_Pg_perc_errors" => ac_pred_Pg_perc_errors,
                # AC and prediction voltage percentage distance
                "ac_pred_Vg_perc_errors" => ac_pred_Vg_perc_errors,
                # DC and prediction generator percentage distance
                "dc_pred_Pg_perc_errors" => dc_pred_Pg_perc_errors,
                # AC and DC max generator percentage distance
                "dc_ac_Pg_perc_errors" => dc_ac_Pg_perc_errors,

                ############################
                # LOADFLOW
                ############################
                # Load flow error w.r.t. predication values, contains pg and vg errin percentage
                "ML_loadflow_pred_perc_errors" => ML_loadflow_pred_perc_errors,
                # Load flow error w.r.t. AC-OPF values, contains pg and vg err in percentage
                "ML_loadflow_acopf_perc_errors" => ML_loadflow_acopf_perc_errors,
                # Load flow error w.r.t. DC values, contains pg and vg errin percentage
                "DC_loadflow_dcopf_perc_errors" => DC_loadflow_dcopf_perc_errors,
                # Load flow error w.r.t. AC-OPF values, contains pg and vg err in percentage
                "DC_loadflow_acopf_perc_errors" => DC_loadflow_acopf_perc_errors,

                ############################
                # OPF COSTS and Time
                ############################
                # DC-OPF, AC-OPF, and Load FLow Dispatch Cost
                "DC_OPF_Cost" => DC_OPF_Cost,
                "AC_OPF_Cost" => AC_OPF_Cost,
                "DC_OPF_Time" => DC_OPF_Time,
                "AC_OPF_Time" => AC_OPF_Time,
                # ML_time = 0
                # DC-OPF, AC-OPF, and Load FLow Computation Time (in sec)
                "ML_Load_Flow_Cost" => ML_Load_Flow_Cost,
                "DC_Load_Flow_Cost" => DC_Load_Flow_Cost,
                "ML_Load_Flow_Time" => ML_Load_Flow_Time,
                "DC_Load_Flow_Time" => DC_Load_Flow_Time,

                ############################
                # CONSTRAINT VIOLATIONS
                ############################
                # Voltage and Thermal limit ML_violations (using Nando's function)
                "load_flow_ML_violations" => ML_violations,
                "load_flow_DC_violations" => DC_violations
                )
end
