function save_rt(folder, rt; file::String="VUMPS_rt_env.jld2")
    p = joinpath(folder, file)
    rt_save = Array(rt)
    @info "save a VUMPS runtime environment to $p"
    save(p, "rt", rt_save)
end

function load_rt(folder, atype, ifparallelupdown; file::String="VUMPS_rt_env.jld2")
    p = joinpath(folder, file)
    rt = load(p, "rt")
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
    @info "load a VUMPS runtime environment from $p"
    return rt
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
