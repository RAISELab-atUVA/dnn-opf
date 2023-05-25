using Printf, ArgParse, Statistics, ProgressMeter

""" Parse Arguments """
function parse_dt_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netpath" # do not change
            help = "The path to the input networks (.m)"
            arg_type = String
            default = "data/inputs/"
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "nesta_case14_ieee"
        "--traindata", "-i" # can read pickle files
            help = "The name of the input file, within the netname folder"
            default = "traindata.json"
        "--out-suffix", "-s" ## also used for version
            help = "The suffix given to the output file to identify a given program variant"
            default = nothing
        "--plot-outfile", "-p"
            help = "The name of the result file, within the netname folder"
            default = "losses.png"
        "--results-outfile", "-r"
            help = "The name of the result file, within the netname folder"
            default = "results.pkl"
        "--seed"
            arg_type = Int
            default = 1234
        "--lr"
            help = "The learning rate"
            arg_type = Float64
            default = 0.001
        "--path"
            help = "path to dnn-opf folder"
            default = ""
        "--n_estimators"
            help = "xgb:The number of trees in the ensemble"
            default = 100
            arg_type = Int
        "--eta"
            help = "xgb:The learning rate used to weight each model"
            default = 0.1
            arg_type = Float64
        "--type"
            help = "dt:1, rf:2, xgb:3"
            default = 1
            arg_type = Int
    end
    return parse_args(s)
end

