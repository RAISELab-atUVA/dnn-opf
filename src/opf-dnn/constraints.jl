include("torch.jl")
include("bound_constraint.jl")
include("ohmslaw_constraint.jl")
include("powerbalance_constraint.jl") ## handles multiple generators and loads per bus

mutable struct Constraints
    bnd::Dict
    olc::OhmsLawConstraint
    pbc::PowerBalanceConstraint
    λ::Dict
    #δcost::Dict # difference between this and previous violation
    ρ::Any

    function Constraints(data, datainfo, config, device)
        cbnd = Dict()
        λ = Dict()
        for type in ["pg-bnd", "qg-bnd", "vg-bnd", "vm-bnd", "va-bnd", "flow-bnd"]
            cbnd[type] = BoundConstraint(data, config, type, device)
            λ[type] = 0.0
        end
        ohm = OhmsLawConstraint(datainfo, config, device)
        klc = PowerBalanceConstraint(datainfo, config, device)
        λ["ohm"] = 0.0
        λ["klc"] = 0.0
        new(cbnd, ohm, klc,  λ,  1e-4)
    end
end

function update_λ!(con::Constraints, type, violation)
    # Dual Ascent
    con.λ[type] = con.λ[type] + (con.ρ * violation)
end

function update_allλ!(con::Constraints, violationsEpoch)
    for type in keys(violationsEpoch)
        if type in keys(con.λ)
            con.λ[type] = con.λ[type] + (con.ρ * mean(violationsEpoch[type]))
        end
    end
end

function has_increased(con::Constraints, type) return con.λ[type] > 0 end

function has_decreased(con::Constraints, type) return con.λ[type] <= 0 end


# function update_λ!(con::Constraints, normalize=false)
    # S = sum(abs.(values(con.δcost)))
    # for key in keys(con.λ)
    #     # con.λ[key] = (con.δcost[key] / S)

    #     ## ABS
    #     #con.λ[key] = abs(con.λ[key])
    #     ## WABS
    #     # con.λ[key] = abs(con.λ[key])/S

    #     ## Increase-loss-only  (ILO)
    #     #con.λ[key] = max(0.0, con.λ[key])

    #     ## 1-plus:  if this iteration is worst (<0), increase penality, otherwise decrease
    #     # con.λ[key] = max(0, 1 + con.λ[key])

    #     ## 1-plus-normalized
    #     # con.λ[key] = max(0, 1 + con.λ[key]/S)

    #     # Pascal
    #     #con.λ[key] = con.rho * con.λ[key]
    # end
# end

""" Simple partial function for one specified arg set and a missing one """
function partial(f, a...)
    ( (b...) -> f(a..., b...) )
end
