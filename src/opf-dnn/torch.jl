using PyCall#, Statistics
torch = pyimport("torch")
nn    = pyimport("torch.nn")
F     = pyimport("torch.nn.functional")
optim = pyimport("torch.optim")
np    = pyimport("numpy")
random = pyimport("random")
tu    = pyimport("torch.utils")
#Ten = torch.DoubleTensor
Ten = torch.FloatTensor
iTen = torch.LongTensor
Var = torch.autograd.Variable

function fix_random_params(manual_seed=123)
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # if you are suing GPU
    # torch.cuda.manual_seed(manual_seed)
    # torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.enabled = false
    torch.backends.cudnn.benchmark = false
    torch.backends.cudnn.deterministic = true
end
fix_random_params(1234)

function toarray(tens)
    if tens.requires_grad
        tens = tens.detach()
    end
    if tens.is_cuda
        tens = tens.cpu()
    end
    return tens.numpy()
end
