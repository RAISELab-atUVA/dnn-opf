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

read_C = true
read_noC = true
read_dt = true
read_rf = true
read_xg = true

folder = "$nepoc,$batch" #folder graphs

foldert = "fixed" # trees version
if false
    #name_train = "nesta_case300_ieee"
    nepoc = 5000
    lr = 0.0001
    batch = 200
end
#---------------------------------------------------------------------------------------
#Read dnn with C
if read_C

    err_pg_dnn = Dict()
    err_vm_dnn = Dict()
    err_va_dnn = Dict()
    mean_err_pg_dnn = Array[]
    mean_err_vm_dnn = Float64[]
    mean_err_va_dnn = Float64[]
    pg = Dict() #
    pg_dnn = Dict() #
    #global pd = Float64[]

    pdm = Float64[]
    once_ever = true
    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]

        local filename = "../dnn-opf-main/data/predictionsC/"*name_train*"/$nepoc/dec-"*h*"results-$batch-$lr.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)
        dic_res = content["results"]
        dic_err = dic_res["test_errors"]
        dic_pred = dic_res["predictions"]

        err_pg = Array[]
        err_vm = Float64[]
        err_va = Float64[]
        pgk = Dict() 
        pgk_pred = Dict() #

        once = true

        for i in keys(dic_pred)
            
            temp = dic_pred[i]
            pgt = temp["pg"]
            pred_pgt = temp["pred-pg"]
            pdt = temp["pd"]
            vmt = temp["vm"]
            pred_vmt = temp["pred-vm"]
            vat = temp["va"]
            pred_vat = temp["pred-va"]

            if once 
                for z1 in keys(pgt)
                    push!(err_pg,Float64[])
                    pgk[z1] = Float64[] 
                    pgk_pred[z1] = Float64[]#
                end
                once = false
            end
            if once_ever
                global generators = keys(pgt)
                for z1 in 1:length(keys(pgt))
                    push!(mean_err_pg_dnn,Float64[])
                end
                global once_ever = false
            end

            for i in generators
                push!(pgk[i], pgt[i])
                push!(pgk_pred[i], pred_pgt[i]) #
            end

            for (l,j) in zip(1:length(keys(pgt)),keys(pgt))
                push!(err_pg[l], abs(pgt["$j"]-pred_pgt["$j"]))
            end

            #err vm
            sum_err = 0
            for (i,j) in zip((values(vmt)),values(pred_vmt))
                sum_err+= abs(i - j)
            end
            push!(err_vm, sum_err/length(values(vmt)))
            
            #err va
            sum_err2 = 0
            for (i,j) in zip((values(vat)),values(pred_vat))
                sum_err2+= abs(i - j)
            end
            push!(err_va, sum_err2/length(values(vat)))

            if (h=="1")
                push!(pdm, sum(values(pdt)))
            end
        end
        #global err_pg_dnn["$h"] = Dict("Gen"=>err_pg)
        global err_pg_dnn["$h"] = err_pg
        global err_vm_dnn["$h"] = err_vm
        global err_va_dnn["$h"] = err_va
        pg["$h"] = pgk
        pg_dnn["$h"] = pgk_pred

        for j in 1:length(generators)
            push!(mean_err_pg_dnn[j],mean(mean(err_pg[j])))
        end

        push!(mean_err_vm_dnn,mean(err_vm))
        push!(mean_err_va_dnn,mean(err_va))
    end
    print("DNN w Constraint")
    println(mean(mean_err_pg_dnn))

end

#---------------------------------------------------------------------------------------
#Read dnn w/o
if read_noC

    err_pg_dwo = Dict()
    err_vm_dwo = Dict()
    err_va_dwo = Dict()
    mean_err_pg_dwo = Array[]
    mean_err_vm_dwo = Float64[]
    mean_err_va_dwo = Float64[]
    pg_dwo = Dict()
    #global pd = Float64[]
    
    #pdm = Float64[]
    once_ever = true
    #["1","2","3","4","5"]
    for h in ["1","2","3","4","5"]
    
        local filename = "../dnn-opf-main/data/predictionsnoC/"*name_train*"/$nepoc/dec-"*h*"results-$batch-$lr.pkl"
        infile = open(filename)
        content = Pickle.load(infile)
        close(infile)
        dic_res = content["results"]
        dic_err = dic_res["test_errors"]
        dic_pred = dic_res["predictions"]
    
        err_pg = Array[]
        err_vm = Float64[]
        err_va = Float64[]
        pgk_pred = Dict()

        once = true
    
        for i in keys(dic_pred)
    
            temp = dic_pred[i]
            pgt = temp["pg"]
            pred_pgt = temp["pred-pg"]
            pdt = temp["pd"]
            vmt = temp["vm"]
            pred_vmt = temp["pred-vm"]
            vat = temp["va"]
            pred_vat = temp["pred-va"]
            
            if once 
                for z1 in keys(pgt)
                    push!(err_pg,Float64[])
                    pgk_pred[z1] = Float64[]
                end
                once = false
            end
            if once_ever
                global generators = keys(pgt)
                for z1 in 1:length(keys(pgt))
                    push!(mean_err_pg_dwo,Float64[])
                end
                global once_ever = false
            end
            
            for i in generators
                push!(pgk_pred[i], pred_pgt[i])
            end

            for (l,j) in zip(1:length(keys(pgt)),keys(pgt))
                push!(err_pg[l], abs(pgt["$j"]-pred_pgt["$j"]))
            end
            
            #err vm
            sum_err = 0
            for (i,j) in zip((values(vmt)),values(pred_vmt))
                sum_err+= abs(i - j)
            end
            push!(err_vm, sum_err/length(values(vmt)))
            
            #err vm
            sum_err1 = 0
            for (i,j) in zip((values(vat)),values(pred_vat))
                            sum_err1+= abs(i - j)
            end
            push!(err_va, sum_err1/length(values(vat)))

            if (h=="1")
                #push!(pdm, sum(values(pdt)))
            end
        end
        #global err_pg_dwo["$h"] = Dict("Gen"=>err_pg)
        global err_pg_dwo["$h"] = err_pg
        global err_vm_dwo["$h"] = err_vm
        global err_va_dwo["$h"] = err_va
        pg_dwo["$h"] = pgk_pred

        for j in 1:length(generators)
            push!(mean_err_pg_dwo[j],mean(mean(err_pg[j])))
        end
    
        push!(mean_err_vm_dwo,mean(err_vm))
        push!(mean_err_va_dwo,mean(err_va))
    end
    print("DNN w/o Constraint")
    println(mean(mean_err_pg_dwo))

end
    
#---------------------------------------------------------------------------------------
#Read dt
if read_dt

    file_dt = "data/predictions/" * name_train * "/dt-results-$foldert.pkl"
    pic_new = read_pickle(file_dt)
    #println(data_d["pg"]["pred_pg1"][:,2]) array gen 1
    #println(data_d["pg"]["pred_pg1"][:,2]) array gen 2
    pg_dt = pic_new["pg"]["pred_pg1"]
    pd_dt = pic_new["pd"]["pdm1"]

    #println(pg_dt[1])
    #println(pg_dt[:,1])

    err_pgt = pic_new["err_pg"]
    ferr_pgt = pic_new["f_err_pg"]

    err_vmt = pic_new["err_vm"]
    ferr_vmt = pic_new["f_err_vm"]

    err_vat = pic_new["err_va"]
    ferr_vat = pic_new["f_err_va"]

    mean_err_pgt = pic_new["mean_err_pg"]
    mean_err_vmt = pic_new["mean_err_vm"]
    mean_err_vat = pic_new["mean_err_va"] 

    ohm_dt = pic_new["loss_ohm"]
    klc_dt = pic_new["loss_klc"]

    print("Decision Tree: ")
    println(mean(mean_err_pgt))

end
#---------------------------------------------------------------------------------------
#Read random forest
if read_rf

    filename3 = "data/predictions/" * name_train * "/rf-results-$foldert.pkl"
    pic_new1 = read_pickle(filename3)

    pg_rf = pic_new1["pg"]["pred_pg1"]
    err_pgrf = pic_new1["err_pg"]
    ferr_pgrf = pic_new1["f_err_pg"]

    err_vmrf = pic_new1["err_vm"]
    ferr_vmrf = pic_new1["f_err_vm"]

    err_varf = pic_new1["err_va"]
    ferr_varf = pic_new1["f_err_va"]

    mean_err_pgrf = pic_new1["mean_err_pg"]
    mean_err_vmrf = pic_new1["mean_err_vm"]
    mean_err_varf = pic_new1["mean_err_va"]

    ohm_rf = pic_new1["loss_ohm"]
    klc_rf = pic_new1["loss_klc"]

    print("Random Forest: ")
    println(mean(mean_err_pgrf))
end
#println("Relative error pg: ", (sum_error/length(keys(dic_pred)))*100, "%")

#---------------------------------------------------------------------------------------
#Read xgb
if read_xg

    filename3 = "data/predictions/" * name_train * "/xg-results-$foldert.pkl"
    pic_new1 = read_pickle(filename3)
    
    pg_xg = pic_new1["pg"]["pred_pg1"]
    err_pgxg = pic_new1["err_pg"]
    ferr_pgxg = pic_new1["f_err_pg"]
    
    err_vmxg = pic_new1["err_vm"]
    ferr_vmxg = pic_new1["f_err_vm"]
    
    err_vaxg = pic_new1["err_va"]
    ferr_vaxg = pic_new1["f_err_va"]
    
    mean_err_pgxg = pic_new1["mean_err_pg"]
    mean_err_vmxg = pic_new1["mean_err_vm"]
    mean_err_vaxg = pic_new1["mean_err_va"]
    
    ohm_xg = pic_new1["loss_ohm"]
    klc_xg = pic_new1["loss_klc"]
    print("XGB: ")
    println(mean(mean_err_pgxg))
    end
    #println("Relative error pg: ", (sum_error/length(keys(dic_pred)))*100, "%")

#-------------------------------
#Read constraint violation
ohm_C = Float64[]
ohm_noC = Float64[]
klc_C = Float64[]
klc_noC = Float64[]
for h in ["1","2","3","4","5"]
    local filename = "../dnn-opf-main/data/predictionsC/"*name_train*"/$nepoc/dec-"*h*"results-summary-$batch-$lr.pkl"
    infile = open(filename)
    content = Pickle.load(infile)
    close(infile)
    push!(ohm_C, content["ohm"])
    push!(klc_C, content["klc"])

    local filename11 = "../dnn-opf-main/data/predictionsnoC/"*name_train*"/$nepoc/dec-"*h*"results-summary-$batch-$lr.pkl"
    infile11 = open(filename11)
    content11 = Pickle.load(infile11)
    close(infile11)

    push!(ohm_noC, content11["ohm"])
    push!(klc_noC, content11["klc"])

end
print("C, Ohm, klt: ")
println(mean(ohm_C))
println(mean(klc_C))
print("noC, Ohm, klt: ")
println(mean(ohm_noC))
println(mean(klc_noC))
print("dt, Ohm, klt: ")
println(mean(ohm_dt))
println(mean(klc_dt))
print("rf, Ohm, klt: ")
println(mean(ohm_rf))
println(mean(klc_rf))
print("xg, Ohm, klt: ")
println(mean(ohm_xg))
println(mean(klc_xg))

#----------------------------------BARPLOT---------------------------------
if true
    println("Barplots")
    save_barplot([mean(mean(mean_err_pg_dnn)), mean(mean(mean_err_pg_dwo)), mean(mean_err_pgt), mean(mean_err_pgrf),mean(mean_err_pgxg)],
        ["DNN w Const","DNN w/o Const","Decision Tree", "Random Forest", "XGBoost"],
        "Pg_errors, with mean 5 folds, $name_train ",name_train,folder)

    save_barplot([mean(mean_err_vm_dnn), mean(mean_err_vm_dwo), mean_err_vmt, mean_err_vmrf, mean_err_vmxg],
    ["DNN w Const","DNN w/o Const","Decision Tree", "Random Forest", "XGBoost"],
    "Vm_errors, with mean 5 folds, $name_train ",name_train,folder)

    save_barplot([mean(mean_err_va_dnn), mean(mean_err_va_dwo), mean_err_vat, mean_err_varf, mean_err_vaxg],
    ["DNN w Const","DNN w/o Const","Decision Tree", "Random Forest", "XGBoost"],
    "Va_errors, with mean 5 folds, $name_train ",name_train,folder)

    save_barplot([mean(ohm_C), mean(ohm_noC), ohm_dt, ohm_rf, ohm_xg],
        ["DNN w Const","DNN w/o Const","Decision Tree", "Random Forest", "XGBoost"],
    "Ohm Constraint Violation, $name_train ",name_train,folder)

    save_barplot([mean(klc_C), mean(klc_noC), klc_dt, klc_rf, klc_xg],
    ["DNN w Const","DNN w/o Const","Decision Tree", "Random Forest", "XGBoost"],
    "Klc Constraint Violation, $name_train ",name_train,folder)
end


#-----------------------BOXPLOT----------------------------
if true
    println("Boxplot")
    pgallw = vec(vcat(err_pg_dnn["1"],err_pg_dnn["2"],err_pg_dnn["3"],err_pg_dnn["4"],err_pg_dnn["5"]))
    pgallw1 = Float64[]
    for i in 1:length(pgallw)
        global pgallw1 = vcat(pgallw1,pgallw[i])
    end

    pgallwo = vec(vcat(err_pg_dwo["1"],err_pg_dwo["2"],err_pg_dwo["3"],err_pg_dwo["4"],err_pg_dwo["5"]))
    pgallwo1 = Any[]
    for i in 1:length(pgallwo)
        global pgallwo1 = vcat(pgallwo1,pgallwo[i])
    end

    vmallw = vec(vcat(err_vm_dnn["1"],err_vm_dnn["2"],err_vm_dnn["3"],err_vm_dnn["4"],err_vm_dnn["5"]))
    vmallw1 = Float64[]
    for i in 1:length(vmallw)
        global vmallw1 = vcat(vmallw1,vmallw[i])
    end

    vmallwo = vec(vcat(err_vm_dwo["1"],err_vm_dwo["2"],err_vm_dwo["3"],err_vm_dwo["4"],err_vm_dwo["5"]))
    vmallwo1 = Any[]
    for i in 1:length(vmallwo)
        global vmallwo1 = vcat(vmallwo1,vmallwo[i])
    end

    vaallw = vec(vcat(err_va_dnn["1"],err_va_dnn["2"],err_va_dnn["3"],err_va_dnn["4"],err_va_dnn["5"]))
    vaallw1 = Float64[]
    for i in 1:length(vaallw)
        global vaallw1 = vcat(vaallw1,vaallw[i])
    end

    vaallwo = vec(vcat(err_va_dwo["1"],err_va_dwo["2"],err_va_dwo["3"],err_va_dwo["4"],err_va_dwo["5"]))
    vaallwo1 = Any[]
    for i in 1:length(vaallwo)
        global vaallwo1 = vcat(vaallwo1,vaallwo[i])
    end

    save_boxplot([pgallw1, pgallwo1, ferr_pgt, ferr_pgrf, ferr_vaxg],
        ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
        "Pg_errors, $name_train",name_train,folder)

    save_boxplot([vmallw1, vmallwo1, ferr_vmt, ferr_vmrf, ferr_vaxg],
        ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
        "Vm_errors, $name_train",name_train,folder)

    save_boxplot([vaallw1, vaallwo1, ferr_vat, ferr_varf, ferr_vaxg],
        ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"],
        "Va_errors, $name_train",name_train,folder)
end

#---------------------------PLOT-----------------------
if true
    println("Plots")
    labels = ["Real", "DNN w Const","DNN w/o Const","Decision Tree","Random Forest", "XGBoost"]
    save_plot_pdpg(pdm, pd_dt, pg["1"], pg_dnn["1"], pg_dwo["1"], pg_dt, pg_rf, pg_xg, generators,
    "Pg_per_gen_pd, $name_train",name_train,folder)
end


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


=#