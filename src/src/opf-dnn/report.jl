using PyPlot

function save_plot_losses(losses, config, prefix="", ymax=nothing)
    if ymax == nothing
        ymax = maximum([maximum(losses[i],init=0) for i in keys(losses)])
    end
    ticks= ymax/10#1e-2

    f = plt.figure(figsize=(10,5))
    #xlabel("Iterations"); ylabel("Loss Values")
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    yscale("symlog") # Set the x axis to a logarithmic scale
    subplots_adjust(wspace=0.01) # Set the vertical spacing between axes

    ax1.set_title("Losses")
    ax1.set_yscale("symlog") # Set the x axis to a logarithmic scale
    #ax1.grid("on")
    ax1.set_yticks(0:ticks:ymax) # Set the y-tick range and step size, 0.1 to 0.9 in increments of 0.2
    ax1.set_ylim(0.0,ymax)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    ax1.plot(losses["pg"], linewidth=1.0, linestyle="-", label="pg")
    ax1.plot(losses["va"], linewidth=1.0, linestyle="-", label="va")
    ax1.plot(losses["vm"], linewidth=1.0, linestyle="-", label="vm")
    ax1.plot(losses["ohm"], linewidth=1.0, linestyle="-", label="ohm")
    ax1.plot(losses["klc"], linewidth=1.0, linestyle="-", label="flow-bal")
    ax1.legend()

    ax2.set_title("Losses-Bnd")
    ax2.set_yscale("symlog") # Set the x axis to a logarithmic scale
    #ax2.grid("on")
    ax2.set_yticks(0:ticks:ymax) # Set the y-tick range and step size, 0.1 to 0.9 in increments of 0.2
    ax2.set_ylim(0.0,ymax)
    ax2.get_yaxis().set_visible(false)
    ax2.plot(losses["pg-bnd"], linewidth=1.0, linestyle="-", label="pg-bnd")
    ax2.plot(losses["va-bnd"], linewidth=1.0, linestyle="-", label="va-bnd")
    ax2.plot(losses["vm-bnd"], linewidth=1.0, linestyle="-", label="vm-bnd")
    ax2.plot(losses["flow-bnd"], linewidth=1.0, linestyle="-", label="flow-bnd")
    ax2.legend()

    # fig.canvas.draw() # Update the figure
    suptitle("Losses")
    gcf()

    plt.savefig(get_file_name(config, "plot", prefix))
    plt.close()
end


function save_report!(temp_dict, args, dataset, datainfo, X, Y, Y_pred, Yname, setpoint=false)
    if args["report-verbose"]
        if setpoint
            pd_true, pd_new_true, qd_true, qd_new_true = get_combo_Sd(X, datainfo.n_loads, true)
        else
            pd_true, qd_true = get_Sd_from_pred(X, datainfo, true)
        end

        Y_true = Y.detach().cpu().numpy()
        Y_pred = Y_pred.detach().cpu().numpy()

        for i in 1:args["batchsize"]
            if i <= length(dataset.current_indices)
                dataset_index = dataset.current_indices[i]
                temp_dict[dataset_index] = Dict()
                temp_dict[dataset_index]["pd"] = Dict(datainfo.keys["pd"][k] => v  for (k,v) in enumerate(pd_true[i,:]))
                temp_dict[dataset_index]["qd"] = Dict(datainfo.keys["qd"][k] => v  for (k,v) in enumerate(qd_true[i,:]))
                temp_dict[dataset_index][Yname] = Dict(datainfo.keys[Yname][k] => v  for (k,v) in enumerate(Y_true[i,:]))
                temp_dict[dataset_index]["pred-"*Yname] = Dict(datainfo.keys[Yname][k] => v  for (k,v) in enumerate(Y_pred[i,:]))
                if setpoint
                    temp_dict[dataset_index]["pd_new"] = Dict(datainfo.keys["pd"][k] => v  for (k,v) in enumerate(pd_new_true[i,:]))
                    temp_dict[dataset_index]["qd_new"] = Dict(datainfo.keys["qd"][k] => v  for (k,v) in enumerate(qd_new_true[i,:]))
                end
            end
        end
    end
end

function save_report_flow!(temp_dict, args, dataset, datainfo, X, Y, Y_pred, pname, qname, setpoint=false, suffix="")
    if args["report-verbose"]
        if setpoint
            pd_true, pd_new_true, qd_true, qd_new_true = get_combo_Sd(X, datainfo.n_loads, true)
        else
            pd_true, qd_true = get_Sd_from_pred(X, datainfo, true)
        end

        n = Int(Y.size()[2]/2)
        pij_true =  Y.narrow(1, 0, n).cpu().numpy()
        qij_true =  Y.narrow(1, n, n).cpu().numpy()
        pij_pred =  Y_pred.narrow(1,0,n).detach().cpu().numpy()
        qij_pred =  Y_pred.narrow(1,n,n).detach().cpu().numpy()

        for i in 1:args["batchsize"]
            if i <= length(dataset.current_indices)
               dataset_index = dataset.current_indices[i]
               temp_dict[dataset_index] = Dict()
               temp_dict[dataset_index]["pd"] = Dict(datainfo.keys["pd"][k] => v  for (k,v) in enumerate(pd_true[i,:]))
               temp_dict[dataset_index]["qd"] = Dict(datainfo.keys["qd"][k] => v  for (k,v) in enumerate(qd_true[i,:]))
               temp_dict[dataset_index]["pij"] = Dict(datainfo.keys[pname][k] => v  for (k,v) in enumerate(pij_true[i,:]))
               temp_dict[dataset_index]["pred-pij"*suffix] = Dict(datainfo.keys[pname][k] => v  for (k,v) in enumerate(pij_pred[i,:]))
               temp_dict[dataset_index]["qij"] = Dict(datainfo.keys[qname][k] => v  for (k,v) in enumerate(qij_true[i,:]))
               temp_dict[dataset_index]["pred-qij"*suffix] = Dict(datainfo.keys[qname][k] => v  for (k,v) in enumerate(qij_pred[i,:]))
               if setpoint
                   temp_dict[dataset_index]["pd_new"] = Dict(datainfo.keys["pd"][k] => v  for (k,v) in enumerate(pd_new_true[i,:]))
                   temp_dict[dataset_index]["qd_new"] = Dict(datainfo.keys["qd"][k] => v  for (k,v) in enumerate(qd_new_true[i,:]))
               end
           end
       end
   end
end

function summary_stats(dataset, errors, idx_violations_lb, idx_violations_ub, exp_data)
    return Dict("mean-err" => mean(errors),
                "max-err" => maximum(errors),
                "min-err" => minimum(errors),
                "sd-err" => std(errors),
                "violations" =>
                    Dict(
                    "total-lb" => length(idx_violations_lb),
                    "total-ub" => length(idx_violations_ub),
                    "average-lb" => length(idx_violations_lb) / dataset.n,
                    "average-ub" => length(idx_violations_ub) / dataset.n,
                    "idx-violations-lb" => idx_violations_lb,
                    "idx-violations-ub" => idx_violations_ub,
                    "exp" => exp_data))
end

function summary_stats(dataset, errors, idx_violations, exp_data)
    return Dict("mean-err" => mean(errors),
                "max-err" => maximum(errors),
                "min-err" => minimum(errors),
                "sd-err" => std(errors),
                "violations" =>
                    Dict(
                    "total" => length(idx_violations),
                    "average" => length(idx_violations) / dataset.n,
                    "idx-violations" => idx_violations,
                    "exp" => exp_data))
end
