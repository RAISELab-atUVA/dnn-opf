using PyPlot
using Printf, ProgressMeter, Statistics

import JSON

include("graphs/graphs.jl")
include("opf-dnn/utils.jl")
include("opf-dnn/datautils.jl")
include("opf-dnn/dataloader.jl")
include("opf-dnn/pypickle.jl")

np    = pyimport("numpy")
random = pyimport("random")
plt = pyimport("matplotlib.pyplot")
pad = pyimport("pandas")
skl_t = pyimport("sklearn.tree")
skl_m = pyimport("sklearn.model_selection")

args = parse_commandline()
name_train = args["netname"]
batch = args["batchsize"]
lr = args["lr"]
nepoc = args["nepochs"]

read_C = false
read_noC = false
read_dt = false
read_rf = false
read_xg = false

folder = "fixed"
if true
    #name_train = "nesta_case14_ieee"
    nepoc = 1010
    lr = 0.0001
    batch = 200
end
#---------------------------------------------------------------------------------------
function extract(type)
    load_temp = Dict()

    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]
        if type==1
            filename = "../dnn-opf-main/data/predictionsC/"*name_train*"/$nepoc/dec-"*h*"results-loadflow-$batch-$lr.pkl"
            infile = open(filename)
            content = Pickle.load(infile)
            close(infile)
            dic = content["loadflow"]
        elseif type==2
            filename = "../dnn-opf-main/data/predictionsnoC/"*name_train*"/$nepoc/dec-"*h*"results-loadflow-$batch-$lr.pkl"
            infile = open(filename)
            content = Pickle.load(infile)
            close(infile)
            dic = content["loadflow"]
        elseif type==3
            filename = "../dnn-opf-main/data/predictions/"*name_train*"/dt-results-loadf.pkl"
            infile = open(filename)
            content = Pickle.load(infile)
            close(infile)
            dic = content["loadflow"]["loadflow$h"]
        elseif type==4
            filename = "../dnn-opf-main/data/predictions/"*name_train*"/rf-results-loadf.pkl"
            infile = open(filename)
            content = Pickle.load(infile)
            close(infile)
            dic = content["loadflow"]["loadflow$h"]
        elseif type==5
            filename = "../dnn-opf-main/data/predictions/"*name_train*"/xg-results-loadf.pkl"
            infile = open(filename)
            content = Pickle.load(infile)
            close(infile)
            dic = content["loadflow"]["loadflow$h"]
        end

        ac_pgperc = dic["ac_pred_Pg_perc_errors"]
        ac_vgperc = dic["ac_pred_Vg_perc_errors"]
        ml_viol = dic["load_flow_ML_violations"]
        generator_bounds_a = Float64[]
        voltage_bounds_a = Float64[]
        thermal_limits_a = Float64[]

        ac_pgperc_a = Float64[]
        ac_vgperc_a = Float64[]
        ML_perc_error_pg = Float64[]
        ML_perc_error_vm = Float64[]
        ML_deg_error_va = Float64[]
        ML_error_flow = Float64[]

        ac_cost = dic["AC_OPF_Cost"]
        ml_cost = dic["ML_Load_Flow_Cost"]
        costs = Float64[]

        for i in keys(ac_pgperc)

            generator_bounds = ml_viol[i]["generator_bounds"]
            voltage_bounds = ml_viol[i]["voltage_bounds"]
            thermal_limits = ml_viol[i]["thermal_limits"]

            push!(ac_pgperc_a, ac_pgperc[i]["avg"])
            push!(ac_vgperc_a, ac_vgperc[i]["avg"])
            push!(costs, abs(ac_cost[i]-ml_cost[i]))

            push!(ML_perc_error_pg, dic["ML_perc_error_pg"][i]["avg"])
            push!(ML_perc_error_vm, dic["ML_perc_error_vm"][i]["avg"])
            push!(ML_deg_error_va, dic["ML_deg_error_va"][i]["avg"])
            push!(ML_error_flow, dic["ML_error_flow"][i]["avg"])
            
            temp = Any[]
            for k in keys(generator_bounds)
                push!(temp,generator_bounds[k]["avg"])
            end
            push!(generator_bounds_a,mean(temp))

            temp = Any[]
            for k in keys(voltage_bounds)
                push!(temp,voltage_bounds[k]["avg"])
            end
            push!(voltage_bounds_a,mean(temp))

            temp = Any[]
            for k in keys(thermal_limits)
                push!(temp,thermal_limits[k]["avg"])
            end
            push!(thermal_limits_a,mean(temp))

        end 
        load_temp["$h"] = Dict()
        load_temp["$h"]["ac_pred_Pg_perc_errors"] = ac_pgperc_a
        load_temp["$h"]["ac_pred_Vg_perc_errors"] = ac_vgperc_a

        load_temp["$h"]["ML_perc_error_pg"] = ML_perc_error_pg
        load_temp["$h"]["ML_perc_error_vm"] = ML_perc_error_vm
        load_temp["$h"]["ML_deg_error_va"] = ML_deg_error_va
        load_temp["$h"]["ML_error_flow"] = ML_error_flow

        load_temp["$h"]["generator_bounds"] = generator_bounds_a
        load_temp["$h"]["voltage_bounds"] = voltage_bounds_a
        load_temp["$h"]["thermal_limits"] = thermal_limits_a

        load_temp["$h"]["costs"] = costs
    end
    return load_temp
end

#----
load_dnn = extract(1)
load_dnoc = extract(2)
load_dt = extract(3)
load_rf = extract(4)
load_xg = extract(5)
#------------------------------------------------------------------------------------------


#----------------------------------BARPLOT---------------------------------
#["DNN w Const","DNN w/o Const","Decision Tree", "Random Forest"]
content = "ac_pred_Pg_perc_errors"

save_barplot([mean([mean(load_dnn["1"]["ac_pred_Pg_perc_errors"]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
            mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
            mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
            mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
            mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
    "ac_pred_Pg_perc_errors, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "ac_pred_Vg_perc_errors"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
            mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
            mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
            mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
            mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
    "ac_pred_Vg_perc_errors, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "costs"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
            mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
            mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
            mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
            mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
    "AC_OPF-ML_Costs, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "generator_bounds"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
            mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
            mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
            mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
            mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
    "generator_bounds, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "voltage_bounds"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
"voltage_bounds, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "thermal_limits"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
"thermal_limits, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "ML_perc_error_pg"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
"ML_perc_error_pg, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "ML_perc_error_vm"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
"ML_perc_error_vm, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "ML_deg_error_va"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
"ML_deg_error_va, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "ML_error_flow"
save_barplot([mean([mean(load_dnn["1"][content]),mean(load_dnn["2"][content]),mean(load_dnn["3"][content]),mean(load_dnn["4"][content]),mean(load_dnn["5"][content])]),
mean([mean(load_dnoc["1"][content]),mean(load_dnoc["2"][content]),mean(load_dnoc["3"][content]),mean(load_dnoc["4"][content]),mean(load_dnoc["5"][content])]),
mean([mean(load_dt["1"][content]),mean(load_dt["2"][content]),mean(load_dt["3"][content]),mean(load_dt["4"][content]),mean(load_dt["5"][content])]),
mean([mean(load_rf["1"][content]),mean(load_rf["2"][content]),mean(load_rf["3"][content]),mean(load_rf["4"][content]),mean(load_rf["5"][content])]),
mean([mean(load_xg["1"][content]),mean(load_xg["2"][content]),mean(load_xg["3"][content]),mean(load_xg["4"][content]),mean(load_xg["5"][content])])],
["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
"ML_error_flow, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
#-----------------------BOXPLOT----------------------------
#cat all 5fold results
content = "ac_pred_Pg_perc_errors"
pgallw = vec(vcat(load_dnn["1"][content],load_dnn["2"][content],load_dnn["3"][content],load_dnn["4"][content],load_dnn["5"][content]))
pgallw1 = Float64[]
for i in 1:length(pgallw)
    global pgallw1 = vcat(pgallw1,pgallw[i])
end

pgallwo = vec(vcat(load_dnoc["1"][content],load_dnoc["2"][content],load_dnoc["3"][content],load_dnoc["4"][content],load_dnoc["5"][content]))
pgallwo1 = Any[]
for i in 1:length(pgallwo)
    global pgallwo1 = vcat(pgallwo1,pgallwo[i])
end

pgalldt = vec(vcat(load_dt["1"][content],load_dt["2"][content],load_dt["3"][content],load_dt["4"][content],load_dt["5"][content]))
pgalldt1 = Float64[]
for i in 1:length(pgalldt)
    global pgalldt1 = vcat(pgalldt1,pgalldt[i])
end

pgallrf = vec(vcat(load_rf["1"][content],load_rf["2"][content],load_rf["3"][content],load_rf["4"][content],load_rf["5"][content]))
pgallrf1 = Any[]
for i in 1:length(pgallrf)
    global pgallrf1 = vcat(pgallrf1,pgallrf[i])
end

pgallrf = vec(vcat(load_xg["1"][content],load_xg["2"][content],load_xg["3"][content],load_xg["4"][content],load_xg["5"][content]))
pgallxg1 = Any[]
for i in 1:length(pgallrf)
    global pgallxg1 = vcat(pgallxg1,pgallrf[i])
end

####
content = "ac_pred_Vg_perc_errors"
vmallw = vec(vcat(load_dnn["1"][content],load_dnn["2"][content],load_dnn["3"][content],load_dnn["4"][content],load_dnn["5"][content]))
vmallw1 = Float64[]
for i in 1:length(vmallw)
    global vmallw1 = vcat(vmallw1,vmallw[i])
end

vmallwo = vec(vcat(load_dnoc["1"][content],load_dnoc["2"][content],load_dnoc["3"][content],load_dnoc["4"][content],load_dnoc["5"][content]))
vmallwo1 = Any[]
for i in 1:length(vmallwo)
    global vmallwo1 = vcat(vmallwo1,vmallwo[i])
end

vmalldt = vec(vcat(load_dt["1"][content],load_dt["2"][content],load_dt["3"][content],load_dt["4"][content],load_dt["5"][content]))
vmalldt1 = Float64[]
for i in 1:length(vmalldt)
    global vmalldt1 = vcat(vmalldt1,vmalldt[i])
end

vmallrf = vec(vcat(load_rf["1"][content],load_rf["2"][content],load_rf["3"][content],load_rf["4"][content],load_rf["5"][content]))
vmallrf1 = Any[]
for i in 1:length(vmallrf)
    global vmallrf1 = vcat(vmallrf1,vmallrf[i])
end

pgallrf = vec(vcat(load_xg["1"][content],load_xg["2"][content],load_xg["3"][content],load_xg["4"][content],load_xg["5"][content]))
vmallxg1 = Any[]
for i in 1:length(pgallrf)
    global vmallxg1 = vcat(vmallxg1,pgallrf[i])
end


save_boxplot_dist([pgallw1,pgallwo1,pgalldt1,pgallrf1,pgallxg1],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
    "ac_pred_Pg_perc_errors, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
save_boxplot_dist([vmallw1,vmallwo1,vmalldt1,vmallrf1,vmallxg1],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
    "ac_pred_Vg_perc_errors, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
content = "costs"
vmallw = vec(vcat(load_dnn["1"][content],load_dnn["2"][content],load_dnn["3"][content],load_dnn["4"][content],load_dnn["5"][content]))
vmallw1 = Float64[]
for i in 1:length(vmallw)
    global vmallw1 = vcat(vmallw1,vmallw[i])
end
    
vmallwo = vec(vcat(load_dnoc["1"][content],load_dnoc["2"][content],load_dnoc["3"][content],load_dnoc["4"][content],load_dnoc["5"][content]))
vmallwo1 = Any[]
for i in 1:length(vmallwo)
    global vmallwo1 = vcat(vmallwo1,vmallwo[i])
end
    
vmalldt = vec(vcat(load_dt["1"][content],load_dt["2"][content],load_dt["3"][content],load_dt["4"][content],load_dt["5"][content]))
vmalldt1 = Float64[]
for i in 1:length(vmalldt)
    global vmalldt1 = vcat(vmalldt1,vmalldt[i])
end
    
vmallrf = vec(vcat(load_rf["1"][content],load_rf["2"][content],load_rf["3"][content],load_rf["4"][content],load_rf["5"][content]))
vmallrf1 = Any[]
for i in 1:length(vmallrf)
    global vmallrf1 = vcat(vmallrf1,vmallrf[i])
end

vmallxg = vec(vcat(load_xg["1"][content],load_xg["2"][content],load_xg["3"][content],load_xg["4"][content],load_xg["5"][content]))
vmallxg1 = Any[]
for i in 1:length(vmallrf)
    global vmallxg1 = vcat(vmallxg1,vmallxg[i])
end

save_boxplot_dist([vmallw1,vmallwo1,vmalldt1,vmallrf1, vmallxg1],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest","XGBoost"],
    "AC_OPF-ML_Costs, $name_train, $batch, $lr, $nepoc",name_train,folder)
#
#ac_cost = dic["AC_OPF_Cost"]
#ml_cost = dic["ML_Load_Flow_Cost"]
#=
save_boxplot([values(err_vm),values(err_vmwo),values(ferr_vmt),values(ferr_vmrf)],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest"],
    "Vm_errors, $name_train, $batch, $lr, $nepoc")
=#


#---------------------------PLOT-----------------------

#= 

println("dnn")
save_plot_pg(pdm, d1["Gen1"],d1["Gen2"], "dnn_C_"*name_train)
save_plot_vm(pdm, err_vm, "dnn_C_"*name_train)
println("dnnW")
save_plot_pg(pdm, d2["Gen1"], d2["Gen2"], "dnn_noC_"*name_train)
save_plot_vm(pdm, err_vmwo, "dnn_noC_"*name_train)
println("dt")
save_plot_pgg(pdm, err_pgt, "dt_"*name_train)
save_plot_vm(pdm, err_vmt, "dt_"*name_train)
println("rf")
save_plot_pgg(pdm, err_pgrf, "rf_"*name_train)
save_plot_vm(pdm, err_vmrf, "rf_"*name_train)

=#

# Dict{Any, Any}(29454 => Dict{Any, Any}("pred-vm" => Dict{Any, Any}("4" => 1.0039429664611816, "1" => 1.0602083206176758, "12" => 1.0415884256362915, "2" => 1.036251425743103, "6" => 1.0593717098236084, "11" => 1.0403504371643066, "13" => 1.035583734512329, "5" => 1.0072509050369263, "14" => 1.0139148235321045, "7" => 1.0387648344039917, "8" => 1.0601227283477783, "10" => 1.0296165943145752, "9" => 1.033530831336975, "3" => 1.0032376050949097), "pg" => Dict{Any, Any}("1" => 2.443511486053467, "2" => 0.6299999952316284), "pred-pf" => Dict{Any, Any}("4" => 0.6406006217002869, "1" => 1.6249077320098877, "12" => 0.08957837522029877, "20" => 0.06650814414024353, "2" => 0.8239113092422485, "6" => -0.25286251306533813, "11" => 0.08622732013463974, "13" => 0.203707754611969, "5" => 0.4827253818511963, "15" => 0.3112911283969879, "16" => 0.05539527162909508, "14" => -0.00047565

    #n = length(pgt)
    #n1 = length(pdt)
    #total_pg = sum(pgt[k] for k in keys(pgt))
    #rel_err = broadcast(abs,(values(pgt).-values(pred_pg))./total_pg)
    #push!(ml_pg_err,(mean(rel_err)))
    #=
    for (i,j) in zip((values(pgt)),values(pred_pg))
        push!(pg, i)
        #push!(err_pg, abs(i - j))
    end

pd_dt = content2["pd"]
pg_dt = content2["pg"]
err_pg_dt = content2["err_pg"]
err_vm_dt = content2["err_vm"]
    

function calculate_err_pg(pg, pred_pg)

    err_pg = Float64[]
    for k in 1:length(pg)
       push!(err_pg, (pg[k] - pred_pg[k]).^2)
    end
 
    return err_pg
end

function calculate_err_vm(vm, pred_vm)

    err_vm = Float64[]
 
    for k in 1:length(vm[:,1])
       push!(err_vm, mean((vm[k,:] - pred_vm[k,:]).^2))
    end
 
    return err_vm
end

#Read dnn with C
if read_C

    load_dnn = Dict()

    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]

        local filename = "../dnn-opf-main/data/predictionsC/"*name_train*"/$nepoc/dec-"*h*"results-loadflow-$batch-$lr.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)

        dic = content["loadflow"]
        ac_pgperc = dic["ac_pred_Pg_perc_errors"]
        ac_vgperc = dic["ac_pred_Vg_perc_errors"]

        ml_viol = dic["load_flow_ML_violations"]

        generator_bounds_a = Float64[]
        voltage_bounds_a = Float64[]
        thermal_limits_a = Float64[]

        ac_pgperc_a = Float64[]
        ac_vgperc_a = Float64[]

        ac_cost = dic["AC_OPF_Cost"]
        ml_cost = dic["ML_Load_Flow_Cost"]
        costs = Float64[]

        for i in keys(ac_pgperc)

            generator_bounds = ml_viol[i]["generator_bounds"]
            voltage_bounds = ml_viol[i]["voltage_bounds"]
            thermal_limits = ml_viol[i]["thermal_limits"]

            push!(ac_pgperc_a, ac_pgperc[i]["avg"])
            push!(ac_vgperc_a, ac_vgperc[i]["avg"])

            push!(costs, abs(ac_cost[i]-ml_cost[i]))
            
            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,generator_bounds[i]["avg"])
            end
            push!(generator_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(voltage_bounds)
                push!(temp,voltage_bounds[i]["avg"])
            end
            push!(voltage_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(thermal_limits)
                push!(temp,thermal_limits[i]["avg"])
            end
            push!(thermal_limits_a,mean(temp))

        end 
        load_dnn["$h"] = Dict()
        load_dnn["$h"]["ac_pred_Pg_perc_errors"] = ac_pgperc_a
        load_dnn["$h"]["ac_pred_Vg_perc_errors"] = ac_vgperc_a

        load_dnn["$h"]["generator_bounds"] = generator_bounds_a
        load_dnn["$h"]["voltage_bounds"] = voltage_bounds_a
        load_dnn["$h"]["thermal_limits"] = thermal_limits_a

        load_dnn["$h"]["costs"] = costs
    end

end

#---------------------------------------------------------------------------------------
#Read dnn w/o
if read_noC

    load_dnoc = Dict()

    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]
    
        local filename = "../dnn-opf-main/data/predictionsnoC/"*name_train*"/$nepoc/dec-"*h*"results-loadflow-$batch-$lr.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)
    
        dic = content["loadflow"]
        ac_pgperc = dic["ac_pred_Pg_perc_errors"]
        ac_vgperc = dic["ac_pred_Vg_perc_errors"]

        ac_pgperc_a = Float64[]
        ac_vgperc_a = Float64[]

        ml_viol = dic["load_flow_ML_violations"]
        generator_bounds = ml_viol["generator_bounds"]
        voltage_bounds = ml_viol["voltage_bounds"]
        thermal_limits = ml_viol["thermal_limits"]
        generator_bounds_a = Float64[]
        voltage_bounds_a = Float64[]
        thermal_limits_a = Float64[]
       
        ac_cost = dic["AC_OPF_Cost"]
        ml_cost = dic["ML_Load_Flow_Cost"]
        costs = Float64[]

        for i in keys(ac_pgperc)
            push!(ac_pgperc_a, ac_pgperc[i]["avg"])
            push!(ac_vgperc_a, ac_vgperc[i]["avg"])


            push!(costs, abs(ac_cost[i]-ml_cost[i]))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,generator_bounds[i]["avg"])
            end
            push!(generator_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(voltage_bounds)
                push!(temp,voltage_bounds[i]["avg"])
            end
            push!(voltage_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,thermal_limits[i]["avg"])
            end
            push!(thermal_limits_a,mean(temp))
        end
        load_dnoc["$h"] = Dict()
        load_dnoc["$h"]["ac_pred_Pg_perc_errors"] = ac_pgperc_a
        load_dnoc["$h"]["ac_pred_Vg_perc_errors"] = ac_vgperc_a

        load_dnoc["$h"]["generator_bounds"] = generator_bounds_a
        load_dnoc["$h"]["voltage_bounds"] = voltage_bounds_a
        load_dnoc["$h"]["thermal_limits"] = thermal_limits_a

        load_dnoc["$h"]["costs"] = costs

    end
end
    
#---------------------------------------------------------------------------------------
#Read dt
if read_dt

    load_dt = Dict()

    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]

        local filename = "../dnn-opf-main/data/predictions/"*name_train*"/dt-results-loadf.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)

        dic = content["loadflow"]["loadflow$h"]
        ac_pgperc = dic["ac_pred_Pg_perc_errors"]
        ac_vgperc = dic["ac_pred_Vg_perc_errors"]

        ac_pgperc_a = Float64[]
        ac_vgperc_a = Float64[]

        ml_viol = dic["load_flow_ML_violations"]
        generator_bounds = ml_viol["generator_bounds"]
        voltage_bounds = ml_viol["voltage_bounds"]
        thermal_limits = ml_viol["thermal_limits"]
        generator_bounds_a = Float64[]
        voltage_bounds_a = Float64[]
        thermal_limits_a = Float64[]

        ac_cost = dic["AC_OPF_Cost"]
        ml_cost = dic["ML_Load_Flow_Cost"]
        costs = Float64[]

        for i in keys(ac_pgperc)
            push!(ac_pgperc_a, ac_pgperc[i]["avg"])
            push!(ac_vgperc_a, ac_vgperc[i]["avg"])


            push!(costs, abs(ac_cost[i]-ml_cost[i]))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,generator_bounds[i]["avg"])
            end
            push!(generator_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(voltage_bounds)
                push!(temp,voltage_bounds[i]["avg"])
            end
            push!(voltage_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,thermal_limits[i]["avg"])
            end
            push!(thermal_limits_a,mean(temp))
        end
        load_dt["$h"] = Dict()
        load_dt["$h"]["ac_pred_Pg_perc_errors"] = ac_pgperc_a
        load_dt["$h"]["ac_pred_Vg_perc_errors"] = ac_vgperc_a


        load_dt["$h"]["generator_bounds"] = generator_bounds_a
        load_dt["$h"]["voltage_bounds"] = voltage_bounds_a
        load_dt["$h"]["thermal_limits"] = thermal_limits_a

        load_dt["$h"]["costs"] = costs
    end

end

#---------------------------------------------------------------------------------------
#Read random forest
if read_rf

    load_rf = Dict()

    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]
    
        local filename = "../dnn-opf-main/data/predictions/"*name_train*"/rf-results-loadf.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)
    
        dic = content["loadflow"]["loadflow$h"]
        ac_pgperc = dic["ac_pred_Pg_perc_errors"]
        ac_vgperc = dic["ac_pred_Vg_perc_errors"]

        ac_pgperc_a = Float64[]
        ac_vgperc_a = Float64[]

        ml_viol = dic["load_flow_ML_violations"]
        generator_bounds = ml_viol["generator_bounds"]
        voltage_bounds = ml_viol["voltage_bounds"]
        thermal_limits = ml_viol["thermal_limits"]
        generator_bounds_a = Float64[]
        voltage_bounds_a = Float64[]
        thermal_limits_a = Float64[]
    
        ac_cost = dic["AC_OPF_Cost"]
        ml_cost = dic["ML_Load_Flow_Cost"]
        costs = Float64[]

        for i in keys(ac_pgperc)
            push!(ac_pgperc_a, ac_pgperc[i]["avg"])
            push!(ac_vgperc_a, ac_vgperc[i]["avg"])

            push!(costs, abs(ac_cost[i]-ml_cost[i]))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,generator_bounds[i]["avg"])
            end
            push!(generator_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(voltage_bounds)
                push!(temp,voltage_bounds[i]["avg"])
            end
            push!(voltage_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,thermal_limits[i]["avg"])
            end
            push!(thermal_limits_a,mean(temp))
        end
        load_rf["$h"] = Dict()
        load_rf["$h"]["ac_pred_Pg_perc_errors"] = ac_pgperc_a
        load_rf["$h"]["ac_pred_Vg_perc_errors"] = ac_vgperc_a

        load_rf["$h"]["generator_bounds"] = generator_bounds_a
        load_rf["$h"]["voltage_bounds"] = voltage_bounds_a
        load_rf["$h"]["thermal_limits"] = thermal_limits_a

        load_rf["$h"]["costs"] = costs

    end
end

#---------------------------------------------------------------------------------------
#Read random forest
if read_xg

    load_xg = Dict()

    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]
    
        local filename = "../dnn-opf-main/data/predictions/"*name_train*"/xg-results-loadf.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)
    
        dic = content["loadflow"]["loadflow$h"]
        ac_pgperc = dic["ac_pred_Pg_perc_errors"]
        ac_vgperc = dic["ac_pred_Vg_perc_errors"]

        ac_pgperc_a = Float64[]
        ac_vgperc_a = Float64[]

        ml_viol = dic["load_flow_ML_violations"]
        generator_bounds = ml_viol["generator_bounds"]
        voltage_bounds = ml_viol["voltage_bounds"]
        thermal_limits = ml_viol["thermal_limits"]
        generator_bounds_a = Float64[]
        voltage_bounds_a = Float64[]
        thermal_limits_a = Float64[]
        
        ac_cost = dic["AC_OPF_Cost"]
        ml_cost = dic["ML_Load_Flow_Cost"]
        costs = Float64[]

        for i in keys(ac_pgperc)
            push!(ac_pgperc_a, ac_pgperc[i]["avg"])
            push!(ac_vgperc_a, ac_vgperc[i]["avg"])

            push!(costs, abs(ac_cost[i]-ml_cost[i]))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,generator_bounds[i]["avg"])
            end
            push!(generator_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(voltage_bounds)
                push!(temp,voltage_bounds[i]["avg"])
            end
            push!(voltage_bounds_a,mean(temp))

            temp = Any[]
            for i in keys(generator_bounds)
                push!(temp,thermal_limits[i]["avg"])
            end
            push!(thermal_limits_a,mean(temp))
        end
        load_xg["$h"] = Dict()
        load_xg["$h"]["ac_pred_Pg_perc_errors"] = ac_pgperc_a
        load_xg["$h"]["ac_pred_Vg_perc_errors"] = ac_vgperc_a

        load_xg["$h"]["generator_bounds"] = generator_bounds_a
        load_xg["$h"]["voltage_bounds"] = voltage_bounds_a
        load_xg["$h"]["thermal_limits"] = thermal_limits_a

        load_xg["$h"]["costs"] = costs

    end
end
=#