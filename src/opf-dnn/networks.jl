include("torch.jl")

"""
    3-tailed networks with:

                      --> output 1:  (vm)  (voltage for each bus)
    input (pd, qd) == --> output 2:  (va)  (voltage angle per bus)
                      --> output 3:  (pg)  (generator active power)
                     [--> output 4:  (qg)  (generator reactive power)] (not used)
"""
@pydef mutable struct DeepFFNetPV <: nn.Module
    function __init__(self, indim, outdim, nettype)
        pybuiltin(:super)(DeepFFNetPV, self).__init__()
        self.af = nn.ReLU

        _iSd = nettype == "dec" ? Int(indim["Sd"]*2) : indim["Sd"]
        (i_vm, i_va, i_pg) = (_iSd + indim["vm"], _iSd + indim["va"], _iSd + indim["pg"])
        (o_vm, o_va, o_pg) = (outdim["vm"], outdim["va"], outdim["pg"])

        h1_vm = nettype == "dec" ? Int(i_vm*2) : i_vm
        h2_vm = nettype == "dec" ? Int(o_vm*2) : o_vm
        h1_va = nettype == "dec" ? Int(i_va*2) : i_va
        h2_va = nettype == "dec" ? Int(o_va*2) : o_va
        h1_pg = nettype == "dec" ? Int(i_pg*2) : i_pg
        h2_pg = nettype == "dec" ? Int(o_pg*2) : o_pg

        self.layersIn = nn.Sequential(nn.Linear(indim["Sd"], _iSd),
                                    self.af(),
                                    nn.Linear(_iSd, _iSd),
                                    self.af())

        self.layersVm = nn.Sequential(nn.Linear(i_vm, h1_vm),
                                    self.af(),
                                    nn.Linear(h1_vm, h1_vm),
                                    self.af(),
                                    nn.Linear(h1_vm, h2_vm),
                                    self.af(),
                                    nn.Linear(h2_vm, o_vm))

        self.layersVa = nn.Sequential(nn.Linear(i_va, h1_va),
                                    self.af(),
                                    nn.Linear(h1_va, h1_va),
                                    self.af(),
                                    nn.Linear(h1_va, h2_va),
                                    self.af(),
                                    nn.Linear(h2_va, o_va))

        self.layersPg = nn.Sequential(nn.Linear(i_pg, h1_pg),
                                    self.af(),
                                    nn.Linear(h1_pg, h1_pg),
                                    self.af(),
                                    nn.Linear(h1_pg, h2_pg),
                                    self.af(),
                                    nn.Linear(h2_pg, o_pg))
    end

    function forward(self, inS, inVm, inVa, inPg)
        h1 = self.layersIn(inS)
        i_vm = torch.cat((h1, inVm), 1)
        i_va = torch.cat((h1, inVa), 1)
        i_pg = torch.cat((h1, inPg), 1)
        o_vm = self.layersVm(i_vm)  # vm
        o_va = self.layersVa(i_va)  # va
        o_pg = self.layersPg(i_pg)  # pg
        return (o_vm, o_va, o_pg)
    end
end     # module


@pydef mutable struct DeepFFNetPVNaive <: nn.Module
    function __init__(self, indim, outdim, nettype)
        pybuiltin(:super)(DeepFFNetPVNaive, self).__init__()
        self.af = nn.ReLU

        _iSd = nettype == "dec" ? Int(indim["Sd"]*2) : indim["Sd"]
        i_vm = i_va = i_pg = _iSd
        (o_vm, o_va, o_pg) = (outdim["vm"], outdim["va"], outdim["pg"])

        h1_vm = nettype == "dec" ? Int(i_vm*2) : i_vm
        h2_vm = nettype == "dec" ? Int(o_vm*2) : o_vm
        h1_va = nettype == "dec" ? Int(i_va*2) : i_va
        h2_va = nettype == "dec" ? Int(o_va*2) : o_va
        h1_pg = nettype == "dec" ? Int(i_pg*2) : i_pg
        h2_pg = nettype == "dec" ? Int(o_pg*2) : o_pg

        self.layersIn = nn.Sequential(nn.Linear(indim["Sd"], _iSd),
                                    self.af(),
                                    nn.Linear(_iSd, _iSd),
                                    self.af())

        self.layersVm = nn.Sequential(nn.Linear(i_vm, h1_vm),
                                    self.af(),
                                    nn.Linear(h1_vm, h1_vm),
                                    self.af(),
                                    nn.Linear(h1_vm, h2_vm),
                                    self.af(),
                                    nn.Linear(h2_vm, o_vm))

        self.layersVa = nn.Sequential(nn.Linear(i_va, h1_va),
                                    self.af(),
                                    nn.Linear(h1_va, h1_va),
                                    self.af(),
                                    nn.Linear(h1_va, h2_va),
                                    self.af(),
                                    nn.Linear(h2_va, o_va))

        self.layersPg = nn.Sequential(nn.Linear(i_pg, h1_pg),
                                    self.af(),
                                    nn.Linear(h1_pg, h1_pg),
                                    self.af(),
                                    nn.Linear(h1_pg, h2_pg),
                                    self.af(),
                                    nn.Linear(h2_pg, o_pg))
    end

    function forward(self, inS)
        h1 = self.layersIn(inS)
        o_vm = self.layersVm(h1)  # vm
        o_va = self.layersVa(h1)  # va
        o_pg = self.layersPg(h1)  # pg
        return (o_vm, o_va, o_pg)
    end
end     # module


# takes in a module and applies the specified weight initialization
function weights_init_uniform_rule(model)
    # for every Linear layer in a model..
    for m in model.layers
        layer = m.__class__.__name__
        if layer == "Linear"
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
        end
    end
end
