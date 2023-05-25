function check_eq(a, b, TOL=1e-6)
    return abs(a - b) < TOL
end

function sum0(vec)
    return length(vec) > 0 ? sum(vec) : 0.0
end

""" Generator Bounds """
function check_generator_bounds(pm, sol, nw=0, cnd=1, TOL=1e-6)
    violations = Dict("pmin" => Dict("n" => 0, "avg" => 0.0),
                      "pmax" => Dict("n" => 0, "avg" => 0.0),
                      "qmin" => Dict("n" => 0, "avg" => 0.0),
                      "qmax" => Dict("n" => 0, "avg" => 0.0))
    avg = Dict("pmin" => [], "pmax" => [], "qmin" => [], "qmax" => [], "diff" => [])

    for (k, ori_gen) in ref(pm, :gen)
        gen = sol["solution"]["gen"][string(k)]
        push!(avg["diff"], abs(abs(gen["pg"]) - abs(ori_gen["pg"])))
        if !(gen["pg"] >= ori_gen["pmin"] - TOL)
            violations["pmin"]["n"] += 1
            push!(avg["pmin"], abs(abs(gen["pg"]) - abs(ori_gen["pmin"])))
        end
        if !(gen["pg"] <= ori_gen["pmax"] + TOL)
            violations["pmax"]["n"] += 1
            push!(avg["pmax"], abs(abs(gen["pg"]) - abs(ori_gen["pmax"])))
        end
        if !(gen["qg"] >= ori_gen["qmin"] - TOL)
            violations["qmin"]["n"] += 1
            push!(avg["qmin"], abs(abs(gen["qg"]) - abs(ori_gen["qmin"])))
        end
        if !(gen["qg"] <= ori_gen["qmax"] + TOL)
            violations["qmax"]["n"] += 1
            push!(avg["qmax"], abs(abs(gen["qg"]) - abs(ori_gen["qmax"])))
        end
    end
    for k in keys(violations)
        violations[k]["avg"] = violations[k]["n"] == 0 ? 0.0 : mean(avg[k])
    end

    return violations, mean(avg["diff"])
end

""" Voltage Bounds """
function check_voltage_bounds(pm, sol, nw=0, cnd=1, TOL=1e-6)
    violations = Dict("vmin" => Dict("n" => 0, "avg" => 0.0),
                      "vmax" => Dict("n" => 0, "avg" => 0.0))
    avg = Dict("vmin" => [], "vmax" => [], "diff" => [])

    for (k, ori_bus) in ref(pm, :bus)
        bus = sol["solution"]["bus"][string(k)]
        push!(avg["diff"], abs(bus["vm"] - ori_bus["vm"]))

        if !(bus["vm"] >= ori_bus["vmin"] - TOL)
            violations["vmin"]["n"] += 1
            push!(avg["vmin"], abs(bus["vm"] - ori_bus["vmin"]))
        end
        if !(bus["vm"] <= ori_bus["vmax"] + TOL)
            violations["vmax"]["n"] += 1
            push!(avg["vmax"], abs(bus["vm"] - ori_bus["vmax"]))
        end
    end
    for k in keys(violations)
        violations[k]["avg"] = violations[k]["n"] == 0 ? 0.0 : mean(avg[k])
    end

    return  violations, mean(avg["diff"])
end

""" Angle Differences """
function check_angle_difference(pm, sol, nw=0, cnd=1, TOL=1e-6)
    violations = Dict("amin" => Dict("n" => 0, "avg" => 0.0),
                      "amax" => Dict("n" => 0, "avg" => 0.0))
    avg = Dict("amin" => [], "amax" => [], "diff" => [])

    for i in ids(pm, :branch)
        branch = ref(pm, nw, :branch, i)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        pair = (f_bus, t_bus)
        buspair = ref(pm, nw, :buspairs, pair)

        if buspair["branch"] == i
            angmin = buspair["angmin"][cnd]
            angmax = buspair["angmax"][cnd]
            va_fr = sol["solution"]["bus"][string(f_bus)]["va"]
            va_to = sol["solution"]["bus"][string(t_bus)]["va"]

            if !(va_fr - va_to <= angmax + TOL)
                violations["amax"]["n"] += 1
                push!(avg["amax"], abs(va_fr - va_to - angmax))
            end
            if !(va_fr - va_to >= angmin - TOL)
                violations["amax"]["n"] += 1
                push!(avg["amax"], abs(va_fr - va_to - angmin))
            end
        end
    end
    for k in keys(violations)
        violations[k]["avg"] = violations[k]["n"] == 0 ? 0.0 : mean(avg[k])
    end

    return  violations
end

""" Thermal Limits """
function check_thermal_limits(pm, sol, nw=0, cnd=1, TOL=1e-6)
    violations = Dict("Sf" => Dict("n" => 0, "avg" => 0.0),
                      "St" => Dict("n" => 0, "avg" => 0.0))
    avg = Dict("Sf" => [], "St" => [], "diff" => [])

    for (k, ori_line) in ref(pm, :branch)
        line = sol["solution"]["branch"][string(k)]
        Sf = line["pf"]^2 + line["qf"]^2
        St = line["pt"]^2 + line["qt"]^2
        rate = ori_line["rate_a"]

        if !(Sf <= rate^2 + TOL)
            violations["Sf"]["n"] += 1
            push!(avg["Sf"], Float64(abs(Sf - rate^2)))
        end
        if !(St <= rate^2 + TOL)
            violations["St"]["n"] += 1
            push!(avg["St"], Float64(abs(Sf - rate^2)))
        end
    end
    for k in keys(violations)
        violations[k]["avg"] = violations[k]["n"] == 0 ? 0.0 : mean(avg[k])
    end
    return violations
end

""" Ohms Lows """
function check_ohms_law(pm, sol, nw=0, cnd=1, TOL=1e-6)
    violations = Dict("pf" => Dict("n" => 0, "avg" => 0.0),
                      "qf" => Dict("n" => 0, "avg" => 0.0),
                      "pt" => Dict("n" => 0, "avg" => 0.0),
                      "qt" => Dict("n" => 0, "avg" => 0.0))
    avg = Dict("pf" => [], "qf" => [], "pt" => [], "qt" => [])

    for (k, ori_branch) in ref(pm, :branch)
        branch = sol["solution"]["branch"][string(k)]
        f_bus = ori_branch["f_bus"]
        t_bus = ori_branch["t_bus"]
        g, b = calc_branch_y(ori_branch)
        g_fr = ori_branch["g_fr"]
        b_fr = ori_branch["b_fr"]
        g_to = ori_branch["g_to"]
        b_to = ori_branch["b_to"]
        tm = ori_branch["tap"]
        ta = ori_branch["shift"]

        p_fr = branch["pf"]
        q_fr = branch["qf"]
        p_to = branch["pt"]
        q_to = branch["qt"]

        vm_fr = sol["solution"]["bus"][string(f_bus)]["vm"]
        va_fr = sol["solution"]["bus"][string(f_bus)]["va"]
        vm_to = sol["solution"]["bus"][string(t_bus)]["vm"]
        va_to = sol["solution"]["bus"][string(t_bus)]["va"]

        p_fr_calc = ((g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta))
        q_fr_calc = (-(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta))
        p_to_calc = ((g+g_to)*vm_to^2 - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta))
        q_to_calc = (-(b+b_to)*vm_to^2 + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta))

        if !check_eq(p_fr, p_fr_calc)
            violations["pf"]["n"] += 1
            #println(abs(p_fr), " \t ", abs(p_fr_calc))
            push!(avg["pf"], Float64(abs(p_fr - p_fr_calc)))
        end
        if !check_eq(q_fr, q_fr_calc)
            violations["qf"]["n"] += 1
            push!(avg["qf"], Float64(abs(q_fr - q_fr_calc)))
        end
        if !check_eq(p_to, p_to_calc)
            violations["pt"]["n"] += 1
            push!(avg["pt"], Float64(abs(p_to - p_to_calc)))
        end
        if !check_eq(q_to, q_to_calc)
            violations["qt"]["n"] += 1
            push!(avg["qt"], Float64(abs(q_to - q_to_calc)))
        end
    end
    for k in keys(violations)
        violations[k]["avg"] = violations[k]["n"] == 0 ? 0.0 : mean(avg[k])
    end
    return violations
end

""" Check Power Balance """
function check_power_balance(pm, sol, nw=0, cnd=1, TOL=1e-6)
    violations = Dict("p" => Dict("n" => 0, "avg" => 0.0),
                      "q" => Dict("n" => 0, "avg" => 0.0))
    avg = Dict("p" => [], "q" => [])

    branches = ref(pm, nw, :branch)
    for (i, bus) in ref(pm, nw, :bus)
        sol_bus = sol["solution"]["bus"][string(i)]
        sol_gen = sol["solution"]["gen"]

        # (l, i, j) where l = arc_ID, i = bus_from, j = bus_to
        bus_arcs = ref(pm, nw, :bus_arcs, i)
        bus_gens = ref(pm, nw, :bus_gens, i)
        bus_loads = ref(pm, nw, :bus_loads, i)
        bus_shunts = ref(pm, nw, :bus_shunts, i)
        bus_pd = [ref(pm, nw, :load, j, "pd", cnd) for j in bus_loads]
        bus_qd = [ref(pm, nw, :load, j, "qd", cnd) for j in bus_loads]
        bus_gs = [ref(pm, nw, :shunt, j, "gs", cnd) for j in bus_shunts]
        bus_bs = [ref(pm, nw, :shunt, j, "bs", cnd) for j in bus_shunts]
        bus_pg = [sol_gen[string(j)]["pg"] for j in bus_gens]
        bus_qg = [sol_gen[string(j)]["qg"] for j in bus_gens]

        vm = sol_bus["vm"] # bus voltage

        pf, qf = [], []
        for (_l, _i, _j) in bus_arcs
            bsol = sol["solution"]["branch"][string(_l)]
            b = branches[_l]
            if b["f_bus"] == _i && b["t_bus"] == _j
                push!(pf, bsol["pf"])
                push!(qf, bsol["qf"])
            else #b["t_bus"] == _i && b["f_bus"] == _j
                push!(pf, bsol["pt"])
                push!(qf, bsol["qt"])
            end
        end

        p_bal_lhs = sum0(pf)
        p_bal_rhs = sum0(bus_pg) - sum0(bus_pd) - sum0(bus_gs) * vm^2
        q_bal_lhs = sum0(qf)
        q_bal_rhs = sum0(bus_qg) - sum0(bus_qd) + sum0(bus_bs) * vm^2

        if !check_eq(p_bal_lhs, p_bal_rhs)
            violations["p"]["n"] += 1
            push!(avg["p"], abs(p_bal_lhs - p_bal_rhs))
        end
        if !check_eq(q_bal_lhs, q_bal_rhs)
            violations["q"]["n"] += 1
            push!(avg["q"], abs(q_bal_lhs - q_bal_rhs))
        end
    end
    for k in keys(violations)
        violations[k]["avg"] = violations[k]["n"] == 0 ? 0.0 : mean(avg[k])
    end
    return violations
end
