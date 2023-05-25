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

#---------------------------------------------------------------------------------------
#Read dnn with C
local filename = "../dnn-opf-main/data/predictionsC/"*name_train*"/$nepoc/dec-"*h*"results-$batch-$lr.pkl"
infilednn = open(filename_dnn)
contentdnn = Pickle.load(infilednn)
close(infilednn)



dic_res = contentdnn["results"]
dic_err = dic_res["test_errors"]
dic_pred = dic_res["predictions"]


dic= dic_pred[15451]

println(dic["pg"])


global ml_pg_err = Float64[]

global total_pg1 = 0
global total_values = 0

global pd = Float64[]
global err_pg = Float64[]
global err_pg2 = Float64[]
global err_vm = Float64[]
global pdm = Float64[]

#Extract pg, pd and vm from results.pkl and calculate errors
for i in keys(dic_pred)

    temp = dic_pred[i]
    pgt = temp["pg"]
    pred_pgt = temp["pred-pg"]
    pdt = temp["pd"]
    vmt = temp["vm"]
    pred_vmt = temp["pred-vm"]

    for (i,j) in zip((values(pgt)),values(pred_pgt))
        push!(err_pg2, abs(i - j))
    end
    
    push!(pdm, sum(values(pdt)))

    sum_err = 0
    for (i,j) in zip((values(vmt)),values(pred_vmt))
        sum_err+= (i - j).^2
    end
    push!(err_vm, sum_err/length(values(vmt)))
end

global cont = 1
for i in err_pg2
    if rem(cont ,2) == 1
        push!(err_pg, i)
    end
    global cont+=1
end
global cont = 1
for i in err_pg2
    if rem(cont ,2) == 0
        push!(err_pg, i)
    end
    global cont+=1
end

#=
#---------------------------------------------------------------------------------------
#Read dt
file_dt = "data/predictions/" * name_train * "/rf-results.pkl"
pic_new = read_pickle(file_dt)
err_pgt = pic_new["err_pg"]
mean_err_pgt = pic_new["mean_err_pg"]
err_vmt = pic_new["err_vm"] 
vm = pic_new["vm"]



println(length(vm))
=#