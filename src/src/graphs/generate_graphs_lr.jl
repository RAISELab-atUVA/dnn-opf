using PyPlot
using Printf, ProgressMeter, Statistics

import JSON

include("graphs.jl")
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

read_C_100 = true
read_noC_100 = true
read_dt = true
read_rf = true
nepoc = 100
all = Float64[]
#---------------------------------------------------------------------------------------
#Read dnn with C
lr = 0.1

err_pg_dnn = Dict()
err_vm_dnn = Dict()
mean_err_pg_dnn = Array[]
mean_err_vm_dnn = Float64[]
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
    once = true

    for i in keys(dic_pred)

        temp = dic_pred[i]
        pgt = temp["pg"]
        pred_pgt = temp["pred-pg"]
        pdt = temp["pd"]
        vmt = temp["vm"]
        pred_vmt = temp["pred-vm"]
        
        if once 
            for z1 in 1:length(keys(pgt))
                push!(err_pg,Float64[])
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

        for (l,j) in zip(1:length(keys(pgt)),keys(pgt))
            push!(err_pg[l], abs(pgt["$j"]-pred_pgt["$j"]))
        end

        #err vm
        sum_err = 0
        for (i,j) in zip((values(vmt)),values(pred_vmt))
            sum_err+= abs(i - j)
        end
        push!(err_vm, sum_err/length(values(vmt)))

        if (h=="1")
            push!(pdm, sum(values(pdt)))
        end
    end
    global err_pg_dnn["$h"] = Dict("Gen"=>err_pg)
    global err_vm_dnn["$h"] = err_vm
    println(length(generators))
    for j in 1:length(generators)
        push!(mean_err_pg_dnn[j],mean(mean(err_pg[j])))
    end

    push!(mean_err_vm_dnn,mean(err_vm))
end
push!(all,mean(mean(mean_err_pg_dnn)))


lr = 0.01
err_pg_dnn = Dict()
err_vm_dnn = Dict()
mean_err_pg_dnn = Array[]
mean_err_vm_dnn = Float64[]
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
    once = true

    for i in keys(dic_pred)

        temp = dic_pred[i]
        pgt = temp["pg"]
        pred_pgt = temp["pred-pg"]
        pdt = temp["pd"]
        vmt = temp["vm"]
        pred_vmt = temp["pred-vm"]
        
        if once 
            for z1 in 1:length(keys(pgt))
                push!(err_pg,Float64[])
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

        for (l,j) in zip(1:length(keys(pgt)),keys(pgt))
            push!(err_pg[l], abs(pgt["$j"]-pred_pgt["$j"]))
        end

        #err vm
        sum_err = 0
        for (i,j) in zip((values(vmt)),values(pred_vmt))
            sum_err+= abs(i - j)
        end
        push!(err_vm, sum_err/length(values(vmt)))

        if (h=="1")
            push!(pdm, sum(values(pdt)))
        end
    end
    global err_pg_dnn["$h"] = Dict("Gen"=>err_pg)
    global err_vm_dnn["$h"] = err_vm
    println(length(generators))
    for j in 1:length(generators)
        push!(mean_err_pg_dnn[j],mean(mean(err_pg[j])))
    end

    push!(mean_err_vm_dnn,mean(err_vm))
end
push!(all,mean(mean(mean_err_pg_dnn)))


lr = 0.001
err_pg_dnn = Dict()
err_vm_dnn = Dict()
mean_err_pg_dnn = Array[]
mean_err_vm_dnn = Float64[]
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
    once = true

    for i in keys(dic_pred)

        temp = dic_pred[i]
        pgt = temp["pg"]
        pred_pgt = temp["pred-pg"]
        pdt = temp["pd"]
        vmt = temp["vm"]
        pred_vmt = temp["pred-vm"]
        
        if once 
            for z1 in 1:length(keys(pgt))
                push!(err_pg,Float64[])
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

        for (l,j) in zip(1:length(keys(pgt)),keys(pgt))
            push!(err_pg[l], abs(pgt["$j"]-pred_pgt["$j"]))
        end

        #err vm
        sum_err = 0
        for (i,j) in zip((values(vmt)),values(pred_vmt))
            sum_err+= abs(i - j)
        end
        push!(err_vm, sum_err/length(values(vmt)))

        if (h=="1")
            push!(pdm, sum(values(pdt)))
        end
    end
    global err_pg_dnn["$h"] = Dict("Gen"=>err_pg)
    global err_vm_dnn["$h"] = err_vm
    println(length(generators))
    for j in 1:length(generators)
        push!(mean_err_pg_dnn[j],mean(mean(err_pg[j])))
    end

    push!(mean_err_vm_dnn,mean(err_vm))
end
push!(all,mean(mean(mean_err_pg_dnn)))


lr = 0.0001
nepoc = 1000
err_pg_dnn = Dict()
err_vm_dnn = Dict()
mean_err_pg_dnn = Array[]
mean_err_vm_dnn = Float64[]
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
    once = true

    for i in keys(dic_pred)

        temp = dic_pred[i]
        pgt = temp["pg"]
        pred_pgt = temp["pred-pg"]
        pdt = temp["pd"]
        vmt = temp["vm"]
        pred_vmt = temp["pred-vm"]
        
        if once 
            for z1 in 1:length(keys(pgt))
                push!(err_pg,Float64[])
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

        for (l,j) in zip(1:length(keys(pgt)),keys(pgt))
            push!(err_pg[l], abs(pgt["$j"]-pred_pgt["$j"]))
        end

        #err vm
        sum_err = 0
        for (i,j) in zip((values(vmt)),values(pred_vmt))
            sum_err+= abs(i - j)
        end
        push!(err_vm, sum_err/length(values(vmt)))

        if (h=="1")
            push!(pdm, sum(values(pdt)))
        end
    end
    global err_pg_dnn["$h"] = Dict("Gen"=>err_pg)
    global err_vm_dnn["$h"] = err_vm
    println(length(generators))
    for j in 1:length(generators)
        push!(mean_err_pg_dnn[j],mean(mean(err_pg[j])))
    end

    push!(mean_err_vm_dnn,mean(err_vm))
end
push!(all,mean(mean(mean_err_pg_dnn)))
#---------------------------------------------------------------------------------------


save_barplot(all,
["0.1","0.01","0.001", "0.0001"],
 "Pg_errors, different learning rate values, $name_train, $batch")

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
#-----------------------BOXPLOT----------------------------

#=
save_boxplot([values(err_pg_dnn["1"]),values(err_pg_dwo["1"]),err_pgt,err_pgrf],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest"],
    "Pg_errors, $name_train, $batch, $lr, $nepoc")

save_boxplot([err_vm,err_vmwo,err_vmt,err_vmrf],
    ["DNN w Const","DNN w/o Const","Decision Tree","Random Forest"],
    "Vm_errors, $name_train, $batch, $lr, $nepoc")
=#
#----------------------------------BARPLOT---------------------------------
#println(err_pg)



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