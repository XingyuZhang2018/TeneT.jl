module OracleGateRunner

using TOML

const DEFAULT_GATES_FILE = normpath(joinpath(@__DIR__, "..", "oracles", "gates.toml"))
const DEFAULT_ROOT = normpath(joinpath(@__DIR__, ".."))

struct Gate
    name::String
    description::String
    cmd::Vector{String}
    groups::Set{String}
    requires_mpi::Bool
    requires_gpu::Bool
    requires_cluster::Bool
end

function _get_bool(entry, key)
    value = get(entry, key, false)
    value isa Bool || throw(ArgumentError("gate field `$key` must be a Bool"))
    return value
end

function _get_strings(entry, key; required::Bool = true)
    value = get(entry, key, nothing)
    if value === nothing
        required && throw(ArgumentError("missing required gate field `$key`"))
        return String[]
    end
    value isa Vector || throw(ArgumentError("gate field `$key` must be an array of strings"))
    all(x -> x isa AbstractString, value) ||
        throw(ArgumentError("gate field `$key` must be an array of strings"))
    return String.(value)
end

function load_gates(path::AbstractString = DEFAULT_GATES_FILE)
    raw = TOML.parsefile(path)
    raw_gates = get(raw, "gate", nothing)
    raw_gates isa AbstractDict || throw(ArgumentError("expected [gate.<name>] tables in $path"))

    gates = Dict{String, Gate}()
    for name in sort!(collect(keys(raw_gates)))
        entry = raw_gates[name]
        entry isa AbstractDict || throw(ArgumentError("gate `$name` must be a TOML table"))
        description = get(entry, "description", "")
        description isa AbstractString ||
            throw(ArgumentError("gate `$name` field `description` must be a string"))
        cmd = _get_strings(entry, "cmd")
        isempty(cmd) && throw(ArgumentError("gate `$name` field `cmd` must not be empty"))
        groups = Set(_get_strings(entry, "groups"))
        isempty(groups) && throw(ArgumentError("gate `$name` must have at least one group"))
        gates[String(name)] = Gate(
            String(name),
            String(description),
            cmd,
            groups,
            _get_bool(entry, "requires_mpi"),
            _get_bool(entry, "requires_gpu"),
            _get_bool(entry, "requires_cluster"),
        )
    end
    return gates
end

function _quote_arg(arg::AbstractString)
    isempty(arg) && return "\"\""
    occursin(r"\s", arg) || return arg
    return repr(String(arg))
end

command_string(gate::Gate) = join(_quote_arg.(gate.cmd), " ")

function _requirements(gate::Gate)
    reqs = String[]
    gate.requires_mpi && push!(reqs, "mpi")
    gate.requires_gpu && push!(reqs, "gpu")
    gate.requires_cluster && push!(reqs, "cluster")
    return isempty(reqs) ? "none" : join(reqs, ",")
end

function print_gate_list(io::IO, gates::AbstractDict{String, Gate})
    for name in sort!(collect(keys(gates)))
        gate = gates[name]
        println(io, name)
        println(io, "  groups: ", join(sort!(collect(gate.groups)), ","))
        println(io, "  requires: ", _requirements(gate))
        println(io, "  cmd: ", command_string(gate))
        !isempty(gate.description) && println(io, "  ", gate.description)
    end
    return nothing
end

function _check_known_names(gates, names)
    unknown = filter(name -> !haskey(gates, name), names)
    isempty(unknown) || throw(ArgumentError("unknown gate(s): $(join(unknown, ", "))"))
    return nothing
end

function select_gate_names(gates::AbstractDict{String, Gate};
                           names::Vector{String} = String[],
                           groups::Vector{String} = String[],
                           allow_cluster::Bool = false)
    selected = String[]
    if !isempty(names)
        _check_known_names(gates, names)
        append!(selected, names)
    end
    if !isempty(groups)
        wanted = Set(groups)
        append!(selected, [name for (name, gate) in gates if !isempty(intersect(gate.groups, wanted))])
    end
    isempty(selected) && append!(selected, sort!(collect(keys(gates))))

    selected = sort!(unique(selected))
    blocked = [name for name in selected if gates[name].requires_cluster && !allow_cluster]
    isempty(blocked) || throw(ArgumentError(
        "refusing cluster gate(s) without --allow-cluster: $(join(blocked, ", "))"
    ))
    return selected
end

function run_gate(gate::Gate; root::AbstractString = DEFAULT_ROOT, dry_run::Bool = false, io::IO = stdout)
    println(io, "==> ", gate.name)
    println(io, "    ", command_string(gate))
    dry_run && return true
    return success(Cmd(Cmd(gate.cmd); dir = root))
end

function run_gates(gates::AbstractDict{String, Gate}, names::Vector{String};
                   root::AbstractString = DEFAULT_ROOT,
                   dry_run::Bool = false,
                   io::IO = stdout)
    ok = true
    for name in names
        ok &= run_gate(gates[name]; root, dry_run, io)
    end
    return ok
end

mutable struct Options
    list::Bool
    dry_run::Bool
    allow_cluster::Bool
    gates_file::String
    names::Vector{String}
    groups::Vector{String}
end

Options() = Options(false, false, false, DEFAULT_GATES_FILE, String[], String[])

function _need_value(args, i, flag)
    i < length(args) || throw(ArgumentError("$flag requires a value"))
    return args[i + 1], i + 1
end

function parse_args(args::Vector{String})
    opts = Options()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--list"
            opts.list = true
        elseif arg == "--dry-run"
            opts.dry_run = true
        elseif arg == "--allow-cluster"
            opts.allow_cluster = true
        elseif arg == "--gate"
            value, i = _need_value(args, i, arg)
            push!(opts.names, value)
        elseif startswith(arg, "--gate=")
            push!(opts.names, split(arg, "=", limit = 2)[2])
        elseif arg == "--group"
            value, i = _need_value(args, i, arg)
            push!(opts.groups, value)
        elseif startswith(arg, "--group=")
            push!(opts.groups, split(arg, "=", limit = 2)[2])
        elseif arg == "--gates-file"
            value, i = _need_value(args, i, arg)
            opts.gates_file = value
        elseif startswith(arg, "--gates-file=")
            opts.gates_file = split(arg, "=", limit = 2)[2]
        elseif arg == "--help" || arg == "-h"
            throw(ArgumentError("help"))
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
        i += 1
    end
    return opts
end

function print_help(io::IO = stdout)
    println(io, "Usage: julia --project=. scripts/run_oracle_gates.jl [options]")
    println(io)
    println(io, "Options:")
    println(io, "  --list                  List registered oracle gates")
    println(io, "  --group NAME            Run gates in a group (default: quick)")
    println(io, "  --gate NAME             Run one named gate")
    println(io, "  --dry-run               Print commands without executing")
    println(io, "  --allow-cluster         Permit gates marked requires_cluster=true")
    println(io, "  --gates-file PATH       Use an alternate TOML gate registry")
    return nothing
end

function main(args::Vector{String} = ARGS; root::AbstractString = DEFAULT_ROOT, io::IO = stdout)
    opts = try
        parse_args(args)
    catch err
        if err isa ArgumentError && err.msg == "help"
            print_help(io)
            return true
        end
        rethrow()
    end

    gates = load_gates(opts.gates_file)
    if opts.list
        print_gate_list(io, gates)
        return true
    end

    if isempty(opts.names) && isempty(opts.groups)
        push!(opts.groups, "quick")
    end

    names = select_gate_names(gates; names = opts.names, groups = opts.groups,
                              allow_cluster = opts.allow_cluster)
    return run_gates(gates, names; root, dry_run = opts.dry_run, io)
end

end # module OracleGateRunner

if abspath(PROGRAM_FILE) == @__FILE__
    try
        OracleGateRunner.main() || exit(1)
    catch err
        if err isa ArgumentError
            println(stderr, "error: ", err.msg)
            OracleGateRunner.print_help(stderr)
            exit(2)
        end
        rethrow()
    end
end
