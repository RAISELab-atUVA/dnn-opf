using PyPlot
using Printf, ProgressMeter, Statistics
using JuMP 
using Ipopt
using Random

import Pickle
import JSON

#change .png to .svg or .pdf for better graphs in latex
function save_plot_pdpg(pd,pd_t, pg, pg_dnn, pg_dwo,
    pg_dt, pg_rf, pg_xg, generators, post, netname, folder)

    plt.clf()

    j = 1
    for i in generators

        plt.clf()

        plt.ylabel("Pg")
        plt.xlabel("Total pd")
        plt.title(post *", Gen: $i",y=1.08)

        println("$(@sprintf("Test Errors: max: %.6f min: %.6f avg: %.6f ",
            maximum(pg[i]), minimum(pg[i]), mean(pg[i]) ))")
        println("$(@sprintf("Test Errors: max: %.6f min: %.6f avg: %.6f ",
            maximum(pg_dt[:,j]), minimum(pg_dt[:,j]), mean(pg_dt[:,j]) ))")
        #

        println(sortperm(pd)==sortperm(pd_dt))

        println("og og_dt pred_dt pred_rf")
        #println(length(pg[i])) #6560
        #println(length(pd_dt)) #7274

        og = hcat(pd,pg[i])
        dnn = hcat(pd,pg_dnn[i])
        dwo = hcat(pd,pg_dwo[i])
        dt = hcat(pd_dt,pg_dt[:,j])     
        rf = hcat(pd_dt,pg_rf[:,j])
        xg = hcat(pd_dt,pg_xg[:,j])

        og = og[sortperm(og[:, 1]), :]
        dnn = dnn[sortperm(dnn[:, 1]), :]
        dwo = dwo[sortperm(dwo[:, 1]), :]
        dt = dt[sortperm(dt[:, 1]), :]
        rf = rf[sortperm(rf[:, 1]), :]
        xg = xg[sortperm(xg[:, 1]), :]

        #println(dnn[:,2])
        #println(xg[:,2])

        plt.plot(xg[:,1],xg[:,2],linewidth=1, alpha=0.5, linestyle="-", label="XGB")
        plt.plot(rf[:,1],rf[:,2],linewidth=1, alpha=0.5, linestyle="--", label="RF")
        plt.plot(dt[:,1],dt[:,2],linewidth=1, alpha=0.5, linestyle="-.", label="DT")
        plt.plot(dwo[:,1],dwo[:,2],linewidth=1, alpha=0.5, linestyle="-", label="DNN w/o C")
        plt.plot(dnn[:,1],dnn[:,2],linewidth=1, alpha=0.5, linestyle="--", label="DNN")
        plt.plot(og[:,1],og[:,2],linewidth=4, alpha=0.4, linestyle="-.", label="True pg")
        
        err_tot_dnn = 0
        err_tot_dt = 0
        err_tot_rf = 0
        
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(false)

        plt.legend()

        mkpath("$folder/$netname")
        name1 = "$folder/$netname/"* i * "_pg_plot_" * post * ".pdf"
        plt.savefig(name1)
        j+=1
    end
    
    #plt.show()
end

function save_plot_pdpg_single(pd, pg, pg_pred, generators, post, netname)

    plt.clf()

    j = 1
    for i in generators

        plt.clf()

        plt.ylabel("Pg")
        plt.xlabel("Total pd")
        plt.title(post *", Gen: $i",y=1.08)

        #

        og = hcat(pd,pg[i])
        dnn = hcat(pd,pg_pred[i])

        og = og[sortperm(og[:, 1]), :]
        dnn = dnn[sortperm(dnn[:, 1]), :]
        dwo = dwo[sortperm(dwo[:, 1]), :]
        dt = dt[sortperm(dt[:, 1]), :]
        rf = rf[sortperm(rf[:, 1]), :]
        xg = xg[sortperm(xg[:, 1]), :]

        #println(dnn[:,2])
        #println(xg[:,2])

        plt.plot(xg[:,1],xg[:,2],linewidth=0.5, linestyle="-", label="XGB")
        plt.plot(rf[:,1],rf[:,2],linewidth=0.5, linestyle="-", label="RF")
        plt.plot(dt[:,1],dt[:,2],linewidth=0.5, linestyle="-", label="DT")
        plt.plot(dwo[:,1],dwo[:,2],linewidth=0.5, linestyle="-", label="DNN w/o C")
        plt.plot(dnn[:,1],dnn[:,2],linewidth=0.5, linestyle="-", label="DNN")
        plt.plot(og[:,1],og[:,2],linewidth=0.5, linestyle="-", label="True pg")
        
        err_tot_dnn = 0
        err_tot_dt = 0
        err_tot_rf = 0

        for i in 1:length(dnn[:,1])
            err_tot_dnn+=(abs(dnn[i,2]-og[i,2]))
        end

        for i in 1:length(rf[:,1])
            err_tot_rf+=(abs(rf[i,2]-rf[i,2]))
        end

        for i in 1:length(dt[:,1])
            err_tot_dt+=(abs(dt[i,2]-dt[i,2]))
        end

        println(err_tot_dnn/length(og[:,2]))
        println(err_tot_dt/length(og[:,2]))
        println(err_tot_rf/length(og[:,2]))
        
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(false)

        plt.legend()

        mkpath(netname)
        name1 = "$netname/"* i * "_pg_plot_" * post * ".pdf"
        plt.savefig(name1)
        j+=1
    end
    
    #plt.show()
end

function save_boxplot(x, y, post, netname, folder)
    
    plt.clf()

    #ymax = maximum(x)+abs(0.3*maximum(x))
    #ymin = minimum(x)-abs(0.3*maximum(x))

    plt.boxplot(x, labels = y)
    
    plt.yscale("log")
    #plt.axis([ymin, ymax])
    plt.ylabel("Prediction Error")
    plt.title("Pg prediction error " * post,y=1.08)

    mkpath("$folder/$netname")
    name = "$folder/$netname/boxplot_" * post * ".jpg"
    plt.savefig(name)
    #plt.show()
end

function save_boxplot_dist(x,y, post,netname, folder)
    
    plt.clf()

    #ymax = maximum(x)+abs(0.3*maximum(x))
    #ymin = minimum(x)-abs(0.3*maximum(x))

    plt.boxplot(x, labels = y)
    
    plt.yscale("log")
    #plt.axis([ymin, ymax])
    plt.ylabel("Distance % between pred_pg and pg")
    plt.title(post,y=1.08)

    mkpath("$folder/$netname")
    name = "$folder/$netname/boxplot_" * post * ".jpg"
    plt.savefig(name)
    #plt.show()
end

function save_barplot(y1, labels, post, netname, folder)

    plt.clf()

    plt.bar(labels, y1, width=0.5)

    mkpath("$folder/$netname")
    name = "$folder/$netname/barplot_" * post * ".jpg"
    plt.title("Barplot " * post,y=1.08)
    plt.savefig(name)
end



#=

function save_plot_pg(pd, g1,g2, post="")

    plt.clf()

    xmax = maximum(pd)+0.1*maximum(pd)
    xmin = minimum(pd)-0.1*minimum(pd)
    ymin = 0
    if maximum(g1)>maximum(g2)
        ymax = maximum(g1)+0.1*maximum(g1)      
    else ymax = maximum(g2)+0.1*maximum(g2)  
    end

    comb1 = hcat(pd,g1)
    comb2 = hcat(pd,g2)

    sorted1 = comb1[sortperm(comb1[:, 1]), :]
    sorted2 = comb2[sortperm(comb2[:, 1]), :]

    #plt.plot(pd,err_pg[1:nh], label = "Generator 1")
    #plt.plot(pd,err_pg[(nh+1):end], label = "Generator 2")
    plt.plot(sorted1[:,1], sorted1[:,2], label = "Generator 1")
    plt.plot(sorted2[:,1], sorted2[:,2], label = "Generator 2")

    plt.axis([xmin, xmax, ymin, ymax])
    plt.ylabel("Error (pg - pred_pg)")
    plt.xlabel("total pd")
    plt.title("Pd to error " * post)

    plt.legend()
    name1 = "pg_plot_" * post * ".jpg"
    plt.savefig(name1)
    #plt.show()
end

function save_plot_gen(pd, pg, post="")

    plt.clf()

    comb = hcat(pd,pg)


    #plt.plot(pd,err_pg[1:nh], label = "Generator 1")
    #plt.plot(pd,err_pg[(nh+1):end], label = "Generator 2")
    plt.plot(sorted1[:,1], sorted1[:,2], label = "Generator 1")
    plt.plot(sorted2[:,1], sorted2[:,2], label = "Generator 2")

    plt.axis([xmin, xmax, ymin, ymax])
    plt.ylabel("Error (pg - pred_pg)")
    plt.xlabel("total pd")
    plt.title("Pd to error " * post)

    plt.legend()
    name1 = "pg_plot_" * post * ".jpg"
    plt.savefig(name1)
    #plt.show()
end

function save_plot_vm(pd, err_vm, post="")
    
    plt.clf()

    xmax = maximum(pd)
    xmin = minimum(pd)
    ymax = maximum(err_vm) #zoom:.1,.01,.001
    ymin = minimum(err_vm)

    comb1 = hcat(pd,err_vm)
    sorted1 = comb1[sortperm(comb1[:, 1]), :]

    plt.plot(sorted1[:,1], sorted1[:,2], label = "Generators")

    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel("total pd")
    plt.ylabel("mse vm")
    plt.title("pd to vm " * post)

    name = "vm_plot_" * post * ".jpg"
    plt.savefig(name)
    #plt.show()
end

function save_plot_vm_scat(pd, err_vm, post="")
    
    plt.clf()

    xmax = maximum(pd)
    xmin = minimum(pd)
    ymax = maximum(err_vm) #zoom:.1,.01,.001
    ymin = minimum(err_vm)

    plt.scatter(pd,err_vm, label = "Generators")
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel("total pd")
    plt.ylabel("mse vm")
    plt.title("pd to mse vm " * post)

    name = "vm_plot_" * post * ".jpg"
    plt.savefig(name)
    #plt.show()




    for (i, v) in zip(1:length(y), y)
        plt.text(xlocs[i-1] - 0.25, v + 0.01, v)
    end

----------------------------------
    fig, ax = plt.subplots()
    bars = ax.barh(labels, y)
    ax.bar_label(bars)
=#



#=

for (index, value) in zip([1,2,3],y)
        
    plt.text(value, index, "$value")
end
=#

#=
function save_plot_pg_vs(pg_dt, err_pg_dt, pg_dnn, err_pg_dnn, post="")

    ymax = maximum(pg_dt)+0.1*maximum(pg_dt)
    ymin = minimum(pg_dt)-0.1*minimum(pg_dt)
    xmax = maximum(err_pg_dt)+0.1*maximum(err_pg_dt)
    xmin = minimum(err_pg_dt)-0.1*minimum(err_pg_dt)
    xmaxdnn = maximum(err_pg_dnn)+0.1*maximum(err_pg_dnn)
    xmindnn = minimum(err_pg_dnn)-0.1*minimum(err_pg_dnn)

    f = plt.figure(figsize=(10,10))
    xlabel("Error (pg - pred_pg)"); ylabel("Pg")
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    #yscale("symlog") # Set the x axis to a logarithmic scale
    #subplots_adjust(wspace=0.01) # Set the vertical spacing between axes

    ax1.set_title("Pg Decision Tree")
    #ax1.set_yscale("symlog") # Set the x axis to a logarithmic scale
    #ax1.grid("on")
    #ax1.set_yticks(0:ticks:ymax) # Set the y-tick range and step size, 0.1 to 0.9 in increments of 0.2
    ax1.axis([xmin, xmax, ymin, ymax])
    #ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    ax1.scatter(err_pg_dt, pg_dt)
    ax1.legend()


    
    ax2.set_title("Pg Deep Neural Network")
    #ax2.set_yscale("symlog") # Set the x axis to a logarithmic scale
    #ax2.grid("on")
    ax2.axis([xmindnn, xmaxdnn, ymin, ymax])
    #ax2.set_yticks(0:ticks:ymax) # Set the y-tick range and step size, 0.1 to 0.9 in increments of 0.2
    #ax2.get_yaxis().set_visible(false)

    ax2.scatter(err_pg_dnn, pg_dnn)
    ax2.legend()
    

    # fig.canvas.draw() # Update the figure
    #suptitle("Errors")
    gcf()   
    name = "pg_plot_" * post
    plt.savefig(name)
    plt.close()
end
=#