using DecisionTree, MLDataUtils
using PyCall
#using HDF5, JLD


include("opf-dnn/pypickle.jl")
include("opf-dnn/utils.jl")
include("dt-utils.jl")
include("opf-dnn/datautils.jl")
include("opf-dnn/dataloader.jl")
include("opf-dnn/constraints.jl")
include("opf-datagen/restoration_w_hotstart.jl")

np    = pyimport("numpy")
random = pyimport("random")
plt = pyimport("matplotlib.pyplot")
pad = pyimport("pandas")
xg = pyimport("xgboost")
skl_t = pyimport("sklearn.tree")
skl_e = pyimport("sklearn.ensemble")
skl_m = pyimport("sklearn.model_selection")
multi = pyimport("sklearn.multioutput")

#------FUNC--------------------------------------------------------------------------------------------

#Fitting Regressors to the dataset
function dt_train(x_train, y_train)
   type = args["type"]

   if type == 1
      regressor = skl_t.DecisionTreeRegressor()
      regressor.fit(x_train, y_train)
   elseif type == 2
      regressor = skl_e.RandomForestRegressor()
      regressor.fit(x_train, y_train)
   elseif type == 3
      regr = xg.XGBRegressor(n_estimators=args["n_estimators"],
         max_depth=7, eta=args["eta"], subsample=0.7, colsample_bytree=0.8, seed =args["seed"])
      regressor = multi.MultiOutputRegressor(regr).fit(x_train, y_train)
   end

   #regressor = xg.XGBRegressor(objective ="reg:squarederror", n_estimators = 100, seed = 123)
      
   return regressor
end

function dt_solve(x_test, regressor)

   #--------------------Test
   # Predicting the results and comparing them to real values
   y_pred = regressor.predict(x_test)
   
   return y_pred
end

#Calculate pg error over kfold instances
function calculate_err_pg(d_pg, k)

   err_pg = Float64[]
   err_pg_t = Float64[]
   mean_err_pg = Float64[]
   println(length(d_pg["pg1"]))
   for j in 1:length(d_pg["pg1"])

      err_temp = Float64[]

      for i in 1:k
         push!(err_temp, abs(d_pg["pg$i"][j] - d_pg["pred_pg$i"][j]))
         push!(err_pg_t, abs(d_pg["pg$i"][j] - d_pg["pred_pg$i"][j]))
      end
      n= length(err_temp)
      push!(mean_err_pg,mean(err_temp))

      push!(err_pg,abs(d_pg["pg1"][j] - d_pg["pred_pg1"][j]))
   end
   
   return err_pg, mean_err_pg, err_pg_t
end

#Calculate pg error over kfold instances
function calculate_mse_pg(d_pg, k)

   err_pg = Float64[]
   err_pg_t = Float64[]
   mean_err_pg = Float64[]
   for j in 1:length(d_pg["pg1"])

      err_temp = Float64[]

      for i in 1:k
      
         push!(err_temp, (d_pg["pg$i"][j] - d_pg["pred_pg$i"][j]).^2)
         push!(err_pg_t, (d_pg["pg$i"][j] - d_pg["pred_pg$i"][j]).^2)
      end
      n= length(err_temp)
      push!(mean_err_pg,mean(err_temp))

      push!(err_pg,abs(d_pg["pg1"][j] - d_pg["pred_pg1"][j]))
   end
   
   return err_pg, mean_err_pg, err_pg_t
end

#Calculate vm error over kfold instances
function calculate_err_vm(d_vm, k)

   err_vm = Float64[]
   err_vm_t = Float64[]
   temp = Float64[]

   for j in 1:length(d_vm["vm1"][:,1])#7274

      err_temp = Float64[]

      for i in 1:k
      
         push!(err_temp, mean(abs.(d_vm["vm$i"][j,:] - d_vm["pred_vm$i"][j,:])))
         push!(err_vm_t, mean(abs.(d_vm["vm$i"][j,:] - d_vm["pred_vm$i"][j,:])))
      end
      push!(err_vm,mean(err_temp))
      push!(temp,mean(abs.(d_vm["vm1"][j,:] - d_vm["pred_vm1"][j,:])))
   end

   return temp, err_vm, err_vm_t
end

function calculate_err_va(d_vm, k)

   err_vm = Float64[]
   err_va_t = Float64[]
   temp = Float64[]

   for j in 1:length(d_vm["va1"][:,1])#7274

      err_temp = Float64[]

      for i in 1:k
      
         push!(err_temp, mean(abs.(d_vm["va$i"][j,:] - d_vm["pred_va$i"][j,:])))
         push!(err_va_t, mean(abs.(d_vm["va$i"][j,:] - d_vm["pred_va$i"][j,:])))
      end
      push!(err_vm,mean(err_temp))
      push!(temp,mean(abs.(d_vm["va1"][j,:] - d_vm["pred_va1"][j,:])))
   end

   return temp,err_vm, err_va_t
end

#Calculate vm mse error over kfold instances
function calculate_mse_vm(d_pg, k)

   err_pg = Float64[]
   err_pg_t = Float64[]
   mean_err_pg = Float64[]
   for j in 1:length(d_pg["vm1"])

      err_temp = Float64[]

      for i in 1:k
      
         push!(err_temp, (d_pg["vm$i"][j] - d_pg["pred_vm$i"][j]).^2)
         push!(err_pg_t, (d_pg["vm$i"][j] - d_pg["pred_vm$i"][j]).^2)
      end
      n= length(err_temp)
      push!(mean_err_pg,mean(err_temp))

      push!(err_pg,abs(d_pg["vm1"][j] - d_pg["pred_vm1"][j]))
   end
   
   return err_pg, mean_err_pg, err_pg_t
end

#Calculate va mse error over kfold instances
function calculate_mse_va(d_pg, k)

   err_pg = Float64[]
   err_pg_t = Float64[]
   mean_err_pg = Float64[]
   for j in 1:length(d_pg["va1"])

      err_temp = Float64[]

      for i in 1:k
      
         push!(err_temp, (d_pg["va$i"][j] - d_pg["pred_va$i"][j]).^2)
         push!(err_pg_t, (d_pg["va$i"][j] - d_pg["pred_va$i"][j]).^2)
      end
      n= length(err_temp)
      push!(mean_err_pg,mean(err_temp))

      push!(err_pg,abs(d_pg["va1"][j] - d_pg["pred_va1"][j]))
   end
   
   return err_pg, mean_err_pg, err_pg_t
end

#Collect results for loadflow compatibility
function collect_results(sd, vm, va, pg, flow, va_pred, vm_pred, pg_pred, flow_pred, dinfo)
   pd_true, qd_true = get_components_from_Sd(sd, dinfo, true)

   vm_pred, va_pred, pg_pred = toarray(vm_pred), toarray(va_pred), toarray(pg_pred)
   vm_true, va_true, pg_true = toarray(vm), toarray(va), toarray(pg)
   pf_pred, qf_pred, pt_pred, qt_pred = get_flows(flow_pred, true)
   pf_true, qf_true, pt_true, qt_true = get_flows(flow, true)

   batchsize = min(size(sd)[1])
   res_data = Dict()
   for i in 1:batchsize
       #test_index = agent.test_loader.current_indices[i]
       res_data[i] = Dict(
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
   return res_data
end

#--------------------------------------------------------------------------------------
args = parse_dt_commandline()
name_train = args["netname"]

fix_random_params(args["seed"])
Random.seed!(args["seed"])

k = 5 #number of folds


# Read input data, pd, pg, v
data = read_data(args)
datainfo = DataInfo(data, args)
device = torch.device("cpu")


(Sd, Flows, vm, va, pg, qg, data_indexes, pd) = load_naive_datasets(data, datainfo, args)

n_gen = length(pg[1,:])
println(length(pg[1,:]))#2
#X = pd, Y = pg, vm

folds_pd = kfolds(pd,obsdim = :first)
folds_pg = kfolds(pg,obsdim = :first)
folds_vm = kfolds(vm,obsdim = :first)
folds_va = kfolds(va,obsdim = :first)
folds_sd = kfolds(Sd,obsdim = :first)
folds_flow = kfolds(Flows,obsdim = :first)

dic_pd = Dict()
dic_pg = Dict()
dic_pg_train = Dict()
dic_vm = Dict()
dic_va = Dict()

#batchsize = n test for torch code

println(length(folds_pg[1][1]))#

args["batchsize"] = length(folds_pg[1][1])
datainfo = DataInfo(data, args)
constr   = Constraints(data, datainfo, args, device)

datainfo = DataInfo(data, args)
_lkeys = ["vm", "va", "vm-bnd", "va-bnd", "pg", "pg-bnd",
            "ohm", "flow-bnd", "klc"]
_losses = Dict()
results = Dict()

#kfold
for i in 1:5

   train, test, val = DataLoader(Sd, Flows, vm, va, pg, qg,
                                 shuffle=true, batchsize=args["batchsize"], kfold_it=i)
   #
   L = Dict()
   loss = Dict("vm" => F.mse_loss, "va" => F.mse_loss,
            "pg" => F.mse_loss, "qg" => F.mse_loss,
            "ohm" => F.mse_loss,
            "vm-bnd" => partial(bound_penalty, constr.bnd["vm-bnd"]),
            "va-bnd" => partial(bound_penalty, constr.bnd["va-bnd"]),
            "pg-bnd" => partial(bound_penalty, constr.bnd["pg-bnd"]),
            "flow-bnd" => partial(bound_penalty, constr.bnd["flow-bnd"]))
   #
   for (j, (sd_t, flow_t, vm_t, va_t, pg_t, qg_t)) in enumerate(train)
      global regressor_pg = dt_train(sd_t, pg_t)
      global regressor_vm = dt_train(sd_t, vm_t)
      global regressor_va = dt_train(sd_t, va_t)
      global pg_pred_train = dt_solve(sd_t, regressor_pg)

      dic_pg_train["pg$i"] = pg_t
      dic_pg_train["pred_pg$i"] = pg_pred_train
   end

   for (j, (sd_test, flow_test, vm_test, va_test, pg_test, qg_test)) in enumerate(test)

      global pg_pred = dt_solve(sd_test, regressor_pg)
      global vm_pred = dt_solve(sd_test, regressor_vm)
      global va_pred = dt_solve(sd_test, regressor_va)

      dic_pg["pg$i"] = pg_test
      dic_pg["pred_pg$i"] = pg_pred

      dic_vm["vm$i"] = vm_test
      dic_vm["pred_vm$i"] = vm_pred
   
      dic_va["va$i"] = va_test
      dic_va["pred_va$i"] = va_pred
   
      (sdt, flowt, vmt, vat, pgt, qgt, vm_predt, va_predt, pg_predt) = 
         Ten(sd_test).to(device), Ten(flow_test).to(device),
         Ten(vm_test).to(device), Ten(va_test).to(device), 
         Ten(pg_test).to(device), Ten(qg_test).to(device), 
         Ten(vm_pred).to(device), Ten(va_pred).to(device),
         Ten(pg_pred).to(device)

      #""" Predict FLows from va and vm """

      flow_pred = get_Flows(constr.olc, vm_predt, va_predt)

      #oSf, oSt = get_complex_flows(flow_pred)

      #L["flow-bnd"] = loss["flow-bnd"](oSf) + loss["flow-bnd"](oSt)
      #_losses["flow-bnd$i"]= L["flow-bnd"].item()

      L["ohm"] = loss["ohm"](flow_pred, flowt)
      _losses["ohm$i"] =  L["ohm"].item()

      (pdf, qdf) = get_components_from_Sd(sdt, datainfo)
      (pf_pred, _, pt_pred, _) = get_components_from_Sij(flow_pred, datainfo)
      L["klc"] = get_losses(constr.pbc, pdf, pg_predt, vm_predt, pf_pred, pt_pred)
      _losses["klc$i"]= L["klc"].item()

      results["$i"] = collect_results(sdt,vmt,vat,pgt,flowt,
                                 va_predt,vm_predt,pg_predt,flow_pred,
                                 datainfo)

      results["loadflow$i"] = solve_restoration_problem(args, results["$i"], data_indexes)
   end
end

err_pg, mean_err_pg, err_pg_t= calculate_err_pg(dic_pg, k)
println(mean(mean_err_pg))
println(mean(err_pg))
err_pg_train, mean_err_pg_train,_a = calculate_mse_pg(dic_pg_train, k)
err_vm, mean_err_vm,err_vm_t = calculate_err_vm(dic_vm, k)
err_va, mean_err_va,err_va_t = calculate_err_va(dic_va, k)
loss_ohm = mean([_losses["ohm1"],_losses["ohm2"],_losses["ohm3"],_losses["ohm4"],_losses["ohm5"]])
loss_klc = mean([_losses["klc1"],_losses["klc2"],_losses["klc3"],_losses["klc4"],_losses["klc5"]])


#Calculate load sum
global pdm = Float64[]
for k in 1:(length(pd[:,1]))

   push!(pdm, sum(pd[k,:]))

end

#Write data
# pd = power draw, pg = power generated, err_pg = pg - predicted pg
# pdm = total power draw for the instance  
data_d = Dict(
   "pd" => pd, "pdm" =>pdm, 
   "pg" => dic_pg, "err_pg" => err_pg, "mean_err_pg" => mean(mean_err_pg),
   "vm" => dic_vm, "err_vm" => err_vm, "mean_err_vm" => mean(mean_err_vm),
   "va" => dic_va, "err_va" => err_va, "mean_err_va" => mean(mean_err_va),
   "loss_ohm" => loss_ohm, "loss_klc" => loss_klc, 
   "f_loss" => _losses,
   "f_err_pg" => err_pg_t, "f_err_vm" => err_vm_t, "f_err_va" => err_va_t,
   "err_pg_train" => mean(mean_err_pg_train),
   "loadflow" => results
   )

type = args["type"]

if type == 1
   file = "data/predictions/" * name_train * "/dt-results-loadf.pkl"
elseif type == 2
   file = "data/predictions/" * name_train * "/rf-results-loadf.pkl"
elseif type == 3
   file = "data/predictions/" * name_train * "/xg-results-loadf.pkl"
end

println("$(@sprintf("Test Errors: \tvm: %.6f \tva: %.6f \tpg: %.6f \tflow: %.6f \tflow-balance: %.6f",
   mean(mean_err_vm), mean(mean_err_va), mean(mean_err_pg), loss_ohm, loss_klc  ))")

println("Writing: $(file)")
mkpath("data/predictions/" * name_train)
write_pickle(file, data_d)


#-------------------------------------------------------------------CODE DUMP


#save_boxplot(pdm,err_pg,"dt_"*name_train)
#save_plot_pg(pdm, err_pg, "dt_"*name_train)
#save_plot_vm(pdm, err_vm, "dt_"*name_train)

#println(err_pg[1])
#=
x_test,y_test, regressor = dt_train(pd, pg, args)
y_pred = dt_solve(x_test,y_test, regressor)

x_test_v, y_test_v, regressor_v = dt_train(pd, vm, args)
y_pred_v = dt_solve(x_test_v,y_test_v, regressor_v)

pred_vm = y_pred_v
pred_pg = y_pred
pg = y_test
pd = x_test
vm = y_test_v

err_pg = calculate_err_pg(pg, pred_pg)
err_vm = calculate_err_vm(vm, pred_vm)
=#


#score = skl_m.cross_val_score(
#   skl_t.DecisionTreeRegressor(random_state= 42), pd, pg, cv=kf, scoring="neg_mean_squared_error")
#rmse(score.mean())
#println(np.sqrt(-mean(score)))

#df = pad.DataFrame({"Real Values":y_test.reshape(-1), "Predicted Values":y_pred.reshape(-1)})
#println(typeof(x_test))
#println(typeof(y_pred))
#display(y_test)
#display(y_pred)

#pg = y_test[:,1]

#Tuple{LinearAlgebra.Adjoint{Float64, Matrix{Float64}}}
#=
n = length(pg)
global ml_pg_err = Float64[]

total_pg = sum(k for k in pg)
println("total_pg= ",total_pg)
for k in 1:n
   #push!(ml_pg_err,100.0*abs(pg[k] - pred_pg[k])/total_pg)
   push!(ml_pg_err,abs(pg[k] - pred_pg[k])/pg[k]*100)
   push!(err_pg, abs(pg[k] - pred_pg[k]))
end

#println(ml_pg_err)
ML_perc_error_pg = Dict("max" => maximum(ml_pg_err), "avg" => mean(ml_pg_err), "std" => std(ml_pg_err))

println(ML_perc_error_pg)
println("n= ",n)
=#
#=
#Calculate mse vm
global vmm = Float64[]
for k in 1:(length(err_vm[:,1]))

   push!(vmm, mean(err_vm[k,:].^2))

   for k in 1:(length(pd[:,1]))

      push!(pdm, sum(pd[k,:]))
end
=#