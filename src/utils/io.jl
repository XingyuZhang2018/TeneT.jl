function save_rt(folder, rt; file::String)
    p = joinpath(folder, file)
    rt_save = Array(rt)
    @info "save a $(typeof(rt)) environment to $p"
    save(p, "rt", rt_save; iotype=IOStream)
end

function load_rt(folder, atype, ifparallelupdown=false; file::String)
    p = joinpath(folder, file)
    rt = load(p, "rt"; iotype=IOStream)
    if ifparallelupdown
        rtup, rtdown = rt
        @sync begin
            @async begin
                TeneT.set_device_id!(atype, 1)
                rtup = atype(rtup)
            end
            @async begin
                TeneT.set_device_id!(atype, 2)
                rtdown = atype(rtdown)
            end
        end
        rt = (rtup, rtdown)
    else
        rt = atype(rt)
    end
    @info "load a $(typeof(rt)) runtime environment from $p"
    return rt
end

# --------------------------------------------------------------------------- #
#  initialize_env  —  create or load VUMPS boundary environment
# --------------------------------------------------------------------------- #

"""
    initialize_env(A, D, χ, params::iPEPSOptimize; restriction_ipeps=identity)

Create or load the VUMPS runtime environment for boundary contraction.
If `params.ifload_env` is true and a saved environment exists on disk, it is
loaded; otherwise a fresh `VUMPSRuntime` is constructed from the current
iPEPS tensors.

# Arguments
- `A`: raw iPEPS parameter array
- `D`: bond dimension
- `χ`: boundary bond dimension
- `params`: optimization parameters
- `restriction_ipeps`: optional function that enforces symmetry constraints on `A`
"""
function initialize_env(A, D::Int, χ::Int, params::iPEPSOptimize; restriction_ipeps=identity)
    folder_path = joinpath(params.folder, "D$(D)", "environment")
    file_path = joinpath(folder_path, "χ$χ.jld2")

    if hasproperty(params, :ifload_env) && params.ifload_env
        if ispath(file_path)
            try
                return load_rt(folder_path, _arraytype(A), params.boundary_alg.ifparallelupdown; file="χ$χ.jld2")
            catch e
                @warn "Failed to load environment from $file_path: $(sprint(showerror, e)). Creating new environment."
                return _create_new_env(A, χ, params; restriction_ipeps)
            end
        else
            params.verbosity >= 2 && @warn "File $file_path not found. Creating new VUMPS environment."
            return _create_new_env(A, χ, params; restriction_ipeps)
        end
    else
        return _create_new_env(A, χ, params; restriction_ipeps)
    end
end

"""
    _create_new_env(A, χ, params; restriction_ipeps=identity)

Internal helper: build a fresh `VUMPSRuntime` from the iPEPS tensors.
"""
function _create_new_env(A, χ::Int, params::iPEPSOptimize; restriction_ipeps=identity)
    _G_cache[] = nothing
    A = restriction_ipeps(A)
    A = build_A(A, params)
    return init_env(A, χ, params.boundary_alg)
end

"""
    read_last_log(folder, D) -> (i, χ)

Extract the iteration index `i` and environment bond dimension `χ` from the last
line of `folder/D\$(D)/history.log`.

Expected log format:
    i =     6   t = 43039.05 sec    e_χ144 = -0.501858316272094 gnorm = 1.984e-04   Eimag = 1.389e-10

Returns `(i, χ)` as integers; returns `nothing` for any field not found.
"""
function read_last_log(folder::String, D::Int)
    logfile = joinpath(folder, "D$D", "history.log")
    isfile(logfile) || error("Log file not found: $logfile")
    line = readlines(logfile)[end]
    m_i   = match(r"i\s*=\s*(\d+)", line)
    m_chi = match(r"e_χ(\d+)", line)
    last_i   = m_i   === nothing ? nothing : parse(Int, m_i.captures[1])
    last_chi = m_chi === nothing ? nothing : parse(Int, m_chi.captures[1])
    return last_i, last_chi
end

read_last_log(params::iPEPSOptimize, D::Int) = read_last_log(params.folder, D)

# ============================================================================
# Observable log writer
# ============================================================================

"""
    write_obs_log(e, mag, ξ, χ, folder, ::iPEPSOptimize)

Write energy, magnetization, and correlation length to a log file.
"""
function write_obs_log(e, mag, ξ, χ, folder, ::iPEPSOptimize)
    path = joinpath(folder, "observable")
    isdir(path) || mkpath(path)
    obs_log = joinpath(path, "χ$χ.log")
    e_dict = e[2]
    m_dict = mag[2]

    open(obs_log, "w") do io
        write(io, @sprintf("energy_per_site:\n%.15f\n", real(e[1])))

        for bond_type in keys(e_dict)
            write(io, "$bond_type: i j energy\n")
            for pos in keys(e_dict[bond_type])
                write(io, @sprintf("%s %.15f\t", pos, real(e_dict[bond_type][pos])))
            end
            write(io, "\n")
        end

        write(io, @sprintf("magnetization_norm_per_site:\n%.15f\n", real(mag[1])))

        write(io, "magnetization: i j |M| Mx My Mz\n")
        for pos in keys(m_dict)
            write(io, @sprintf("%s %.15f %.15f %.15f %.15f\t", pos, real(m_dict[pos]["|M|"]), real(m_dict[pos]["Mx"]), real(m_dict[pos]["My"]), real(m_dict[pos]["Mz"])))
            write(io, "\n")
        end
        write(io, @sprintf("correlation_length:\n%.15f\n", real(ξ)))
    end
end