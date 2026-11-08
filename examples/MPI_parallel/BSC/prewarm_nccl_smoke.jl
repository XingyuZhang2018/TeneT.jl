const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "..", "..", ".."))
const PROJECT_DIR = joinpath(SCRIPT_DIR, "nccl_smoke_project")
const WRAPPER = joinpath(REPO_ROOT, "src", "contraction", "parallel", "nccl_wrapper.jl")

port = get(ENV, "OFFLINEHPC_PORT", "")
connected = false

if !isempty(port)
    connect_jl = joinpath(SCRIPT_DIR, "connect.jl")
    isfile(connect_jl) || error("OFFLINEHPC_PORT is set, but connect.jl is missing in $(SCRIPT_DIR)")
    include(connect_jl)
    OfflineHPCClient.connect(port=parse(Int, port), check=true)
    connected = true
end

try
    using Pkg
    Pkg.activate(PROJECT_DIR)
    Pkg.instantiate(; allow_autoprecomp=false)

    using MPI
    using CUDA

    ENV["TENET_NCCL_WRAPPER"] = WRAPPER
    include(WRAPPER)

    println("BSC_NCCL_SMOKE_PREWARM_OK")
finally
    if connected
        OfflineHPCClient.disconnect()
    end
end
