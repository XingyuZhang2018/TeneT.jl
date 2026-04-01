# Observable visualization using Makie.jl
# Reads χ*.log files and generates convergence + lattice plots.

using CairoMakie

# ============================================================================
# Log reader: parse χ*.log → (e, mag, ξ, χ)
# ============================================================================

"""
    read_obs_log(logfile) → (e_scalar, e_dict, mag_norm, m_dict, ξ)

Parse a single χ*.log file into the same data structures used by observable().
"""
function read_obs_log(logfile::String)
    lines = readlines(logfile)
    e_scalar = 0.0
    e_dict = Dict{String, Dict{String, Any}}()
    mag_norm = 0.0
    m_dict = Dict{String, Any}()
    ξ = 0.0

    i = 1
    while i <= length(lines)
        line = strip(lines[i])

        if line == "energy_per_site:"
            i += 1
            e_scalar = parse(Float64, strip(lines[i]))
            i += 1
        elseif endswith(line, ": i j energy")
            bond_type = strip(split(line, ":")[1])
            e_dict[bond_type] = Dict{String, Any}()
            i += 1
            # Parse "pos value\tpos value\t..." (may span one line)
            for token in split(strip(lines[i]), '\t')
                token = strip(token)
                isempty(token) && continue
                parts = split(token)
                length(parts) >= 2 || continue
                pos = parts[1]
                val = parse(Float64, parts[2])
                e_dict[bond_type][pos] = val
            end
            i += 1
        elseif line == "magnetization_norm_per_site:"
            i += 1
            mag_norm = parse(Float64, strip(lines[i]))
            i += 1
        elseif startswith(line, "magnetization: i j")
            i += 1
            while i <= length(lines) && !startswith(strip(lines[i]), "correlation_length")
                parts = split(strip(lines[i]))
                if length(parts) >= 5
                    pos = parts[1]
                    m_dict[pos] = Dict{String, Any}(
                        "|M|" => parse(Float64, parts[2]),
                        "Mx"  => parse(Float64, parts[3]),
                        "My"  => parse(Float64, parts[4]),
                        "Mz"  => parse(Float64, parts[5])
                    )
                end
                i += 1
            end
        elseif line == "correlation_length:"
            i += 1
            ξ = parse(Float64, strip(lines[i]))
            i += 1
        else
            i += 1
        end
    end

    return e_scalar, e_dict, mag_norm, m_dict, ξ
end

"""
    read_all_obs_logs(obs_dir) → Vector{(χ, e_scalar, e_dict, mag_norm, m_dict, ξ)}

Scan `obs_dir` for χ*.log files, parse each, return sorted by χ.
"""
function read_all_obs_logs(obs_dir::String)
    results = []
    isdir(obs_dir) || return results
    for f in readdir(obs_dir)
        m = match(r"^χ(\d+)\.log$", f)
        m === nothing && continue
        χ = parse(Int, m.captures[1])
        e_scalar, e_dict, mag_norm, m_dict, ξ = read_obs_log(joinpath(obs_dir, f))
        push!(results, (χ=χ, e=e_scalar, e_dict=e_dict, mag=mag_norm, m_dict=m_dict, ξ=ξ))
    end
    sort!(results; by=r -> r.χ)
    return results
end

# ============================================================================
# Main entry point: read logs → plot
# ============================================================================

"""
    plot_observables(obs_dir, lattice_type, pattern; save_format="png")

Read all χ*.log files in `obs_dir`, generate convergence plot and lattice
plot for the latest χ. Called automatically by observable()/optimise_ipeps()
when `params.ifplot == true`.
"""
function plot_observables(obs_dir::String, lattice_type, pattern::Matrix{Int};
                          save_format::String="png")
    logs = read_all_obs_logs(obs_dir)
    isempty(logs) && return nothing

    # Convergence plot from all logs
    _plot_convergence(obs_dir, logs; save_format)

    # Lattice plot for every χ
    for r in logs
        plot_lattice_obs(r.e_dict, r.m_dict, lattice_type, pattern;
                         save_path=obs_dir, save_format=save_format, χ=r.χ)
    end
    return nothing
end

# ============================================================================
# Convergence plot (reads from log history)
# ============================================================================

function _plot_convergence(obs_dir::String, logs; save_format::String="png")
    n = length(logs)
    χs      = [r.χ for r in logs]
    energies = [r.e for r in logs]
    mags     = [r.mag for r in logs]
    ξs       = [r.ξ for r in logs]

    fig = Figure(size=(700, 900), fontsize=14)

    function _make_panel(pos, ylabel, data, color; title="")
        ax = Axis(fig[pos, 1]; xlabel=(pos == 3 ? "χ" : ""),
                  ylabel=ylabel, title=title,
                  xticks=χs,
                  xgridvisible=true, ygridvisible=true,
                  xgridstyle=:dash, ygridstyle=:dash,
                  xgridcolor=(:black, 0.1), ygridcolor=(:black, 0.1))
        if n == 1
            scatter!(ax, χs, data; color=color, markersize=10)
            yval = data[1]
            margin = max(abs(yval) * 0.1, 1e-6)
            ylims!(ax, yval - margin, yval + margin)
            xlims!(ax, χs[1] - 1, χs[1] + 1)
        else
            lines!(ax, χs, data; color=color, linewidth=2)
            scatter!(ax, χs, data; color=color, markersize=8)
            xpad = max(1, (χs[end] - χs[1]) * 0.05)
            xlims!(ax, χs[1] - xpad, χs[end] + xpad)
        end
        return ax
    end

    _make_panel(1, "Energy / site", energies, :steelblue; title="Observable Convergence")
    _make_panel(2, "|M| mean", mags, :crimson)
    _make_panel(3, "ξ", ξs, :seagreen)

    outfile = joinpath(obs_dir, "convergence.$(save_format)")
    save(outfile, fig; px_per_unit=2)
    return fig
end

# ============================================================================
# Lattice visualization
# ============================================================================

function plot_lattice_obs(e_dict, m_dict, lattice_type, pattern::Matrix{Int};
                          save_path::String, save_format::String="png", χ::Int=0, n_repeat::Int=3)
    isdir(save_path) || mkpath(save_path)

    Ni, Nj = size(pattern)

    unique_sites = Dict{Int, Tuple{Int,Int}}()
    for (key, _) in m_dict
        parts = split(key, ",")
        i, j = parse(Int, parts[1]), parse(Int, parts[2])
        unique_sites[pattern[i, j]] = (i, j)
    end

    all_coords = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    all_mdata  = Dict{Tuple{Int,Int}, Dict{String, Any}}()
    is_original = Dict{Tuple{Int,Int}, Bool}()

    for di in 0:(n_repeat-1), dj in 0:(n_repeat-1)
        for ci in 1:Ni, cj in 1:Nj
            gi = ci + di * Ni
            gj = cj + dj * Nj
            pval = pattern[ci, cj]
            oi, oj = unique_sites[pval]
            all_mdata[(gi, gj)] = m_dict["$oi,$oj"]
            all_coords[(gi, gj)] = _site_xy(lattice_type, gi, gj)
            is_original[(gi, gj)] = (di == 0 && dj == 0)
        end
    end

    all_evals = Float64[]
    for (_, bond_data) in e_dict
        for (_, ev) in bond_data
            push!(all_evals, abs(real(ev)))
        end
    end
    e_max = isempty(all_evals) ? 1.0 : maximum(all_evals)
    e_min = isempty(all_evals) ? 0.0 : minimum(all_evals)

    all_xs = [c[1] for c in values(all_coords)]
    all_ys = [c[2] for c in values(all_coords)]
    xspan = maximum(all_xs) - minimum(all_xs) + 2.0
    yspan = maximum(all_ys) - minimum(all_ys) + 2.0
    fig_size = max(700, round(Int, max(xspan, yspan) * 110))
    fig = Figure(size=(fig_size, fig_size), fontsize=13, backgroundcolor=:white)
    ax = Axis(fig[1, 1]; title="Lattice Observables  (χ=$χ)", aspect=DataAspect(),
              backgroundcolor=:white)
    hidedecorations!(ax)
    hidespines!(ax)

    # Bonds
    _draw_lattice_bonds!(ax, lattice_type, all_coords, all_mdata,
                         e_dict, pattern, unique_sites, Ni, Nj, n_repeat,
                         e_min, e_max)

    # Sites
    for (k, (x, y)) in all_coords
        alpha = is_original[k] ? 1.0 : 0.4
        ms = is_original[k] ? 22 : 16
        scatter!(ax, [x], [y]; color=(:gray70, alpha), markersize=ms,
                 strokewidth=is_original[k] ? 1.5 : 0.5,
                 strokecolor=(:gray40, alpha))
    end

    # Magnetization arrows (original unit cell only)
    mag_max_val = maximum(abs(real(all_mdata[k]["|M|"])) for k in keys(all_coords))
    arrow_scale = mag_max_val > 1e-10 ? 0.45 / mag_max_val : 0.0
    for (k, (x, y)) in all_coords
        is_original[k] || continue
        mdata = all_mdata[k]
        amx = real(mdata["Mx"]) * arrow_scale
        amz = real(mdata["Mz"]) * arrow_scale
        amag = sqrt(amx^2 + amz^2)
        amag < 1e-8 && continue
        arrows!(ax, [x], [y], [amx], [amz];
                color=(:black, 0.9), linewidth=2.5,
                arrowsize=12, arrowcolor=(:black, 0.9))
    end

    # Site labels
    for (k, (x, y)) in all_coords
        is_original[k] || continue
        ci, cj = mod1(k[1], Ni), mod1(k[2], Nj)
        text!(ax, x, y + 0.55; text="($ci,$cj)", fontsize=10, color=:gray20,
              align=(:center, :bottom))
    end

    # Padding
    xmargin = max(1.5, xspan * 0.12)
    ymargin = max(1.5, yspan * 0.12)
    xlims!(ax, minimum(all_xs) - xmargin, maximum(all_xs) + xmargin)
    ylims!(ax, minimum(all_ys) - ymargin, maximum(all_ys) + ymargin)

    # Legend
    legend_entries = []
    for (bond_type, _) in e_dict
        col = _bond_color(bond_type)
        push!(legend_entries, (bond_type, col))
    end
    if !isempty(legend_entries)
        elems = [LineElement(color=col, linewidth=4) for (_, col) in legend_entries]
        labels = [bt for (bt, _) in legend_entries]
        Legend(fig[1, 2], elems, labels; framevisible=false, labelsize=10, patchsize=(20, 4))
    end

    outfile = joinpath(save_path, "lattice_χ$χ.$(save_format)")
    save(outfile, fig; px_per_unit=2)
    return fig
end

# ============================================================================
# Site coordinates
# ============================================================================

function _site_xy(::Honeycomb{:brickwall}, i, j)
    x = (j - 1) * sqrt(3) / 2
    y = -(i - 1) * 1.5 - ((i + j) % 2 == 1 ? 0.5 : 0.0)
    return (x, y)
end

function _site_xy(::Square, i, j)
    return (Float64(j) * 1.5, -Float64(i) * 1.5)
end

function _site_xy(lattice, i, j)
    return (Float64(j) * 1.5, -Float64(i) * 1.5)
end

# ============================================================================
# Bond drawing
# ============================================================================

const _BOND_COLORS = Dict(
    "Jx" => colorant"#00BFFF",
    "Jy" => colorant"#FF8080",
    "Jz" => colorant"#80FF80",
    "J1_Horizontal" => :royalblue, "J1_Vertical" => :forestgreen,
    "J2_Horizontal" => :orange,
    "Diagonal1" => :purple, "Diagonal\\" => :purple,     # \ direction
    "Diagonal2" => :hotpink, "Diagonal/" => :hotpink,    # / direction
)

function _bond_color(bond_type::String)
    for (key, col) in _BOND_COLORS
        occursin(key, bond_type) && return col
    end
    return :gray60
end

function _bond_linewidth(eval, e_min, e_max)
    ae = abs(real(eval))
    if e_max ≈ e_min
        return 14.0
    end
    t = (ae - e_min) / (e_max - e_min)
    return 5.0 + t * 18.0
end

function _draw_lattice_bonds!(ax, ::Honeycomb{:brickwall}, all_coords, all_mdata,
                               e_dict, pattern, unique_sites, Ni, Nj, n_repeat,
                               e_min, e_max)
    pval_positions = Dict{Int, Vector{Tuple{Int,Int}}}()
    for ci in 1:Ni, cj in 1:Nj
        pv = pattern[ci, cj]
        push!(get!(pval_positions, pv, Tuple{Int,Int}[]), (ci, cj))
    end

    for (bond_type, bond_data) in e_dict
        color = _bond_color(bond_type)
        (off1i, off1j), (off2i, off2j) = _bond_offsets_honeycomb(bond_type)
        for (pos_str, eval) in bond_data
            parts = split(pos_str, ",")
            oi, oj = parse(Int, parts[1]), parse(Int, parts[2])
            pv = pattern[oi, oj]
            lw = _bond_linewidth(eval, e_min, e_max)

            for (ci, cj) in pval_positions[pv]
                for di in 0:(n_repeat-1), dj in 0:(n_repeat-1)
                    gi1 = ci + di * Ni + off1i
                    gj1 = cj + dj * Nj + off1j
                    gi2 = ci + di * Ni + off2i
                    gj2 = cj + dj * Nj + off2j

                    haskey(all_coords, (gi1, gj1)) || continue
                    haskey(all_coords, (gi2, gj2)) || continue

                    x1, y1 = all_coords[(gi1, gj1)]
                    x2, y2 = all_coords[(gi2, gj2)]

                    is_orig = (di == 0 && dj == 0)
                    alpha = is_orig ? 0.85 : 0.3
                    lines!(ax, [x1, x2], [y1, y2]; color=(color, alpha),
                           linewidth=lw, linecap=:round)

                    if is_orig
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        elabel = "$(round(real(eval); sigdigits=4))"
                        dx, dy = x2 - x1, y2 - y1
                        blen = sqrt(dx^2 + dy^2)
                        if blen > 1e-6
                            nx, ny = -dy / blen * 0.15, dx / blen * 0.15
                        else
                            nx, ny = 0.1, 0.0
                        end
                        text!(ax, mx + nx, my + ny; text=elabel, fontsize=9,
                              align=(:center, :center), color=:gray30)
                    end
                end
            end
        end
    end
end

function _draw_lattice_bonds!(ax, ::Square, all_coords, all_mdata,
                               e_dict, pattern, unique_sites, Ni, Nj, n_repeat,
                               e_min, e_max)
    pval_positions = Dict{Int, Vector{Tuple{Int,Int}}}()
    for ci in 1:Ni, cj in 1:Nj
        pv = pattern[ci, cj]
        push!(get!(pval_positions, pv, Tuple{Int,Int}[]), (ci, cj))
    end

    for (bond_type, bond_data) in e_dict
        color = _bond_color(bond_type)
        (off1i, off1j), (off2i, off2j) = _bond_offsets_square(bond_type)
        for (pos_str, eval) in bond_data
            parts = split(pos_str, ",")
            oi, oj = parse(Int, parts[1]), parse(Int, parts[2])
            pv = pattern[oi, oj]
            lw = _bond_linewidth(eval, e_min, e_max)

            for (ci, cj) in pval_positions[pv]
                for di in 0:(n_repeat-1), dj in 0:(n_repeat-1)
                    gi1 = ci + di * Ni + off1i
                    gj1 = cj + dj * Nj + off1j
                    gi2 = ci + di * Ni + off2i
                    gj2 = cj + dj * Nj + off2j

                    haskey(all_coords, (gi1, gj1)) || continue
                    haskey(all_coords, (gi2, gj2)) || continue

                    x1, y1 = all_coords[(gi1, gj1)]
                    x2, y2 = all_coords[(gi2, gj2)]
                    is_orig = (di == 0 && dj == 0)
                    alpha = is_orig ? 0.85 : 0.3
                    lines!(ax, [x1, x2], [y1, y2]; color=(color, alpha),
                           linewidth=lw, linecap=:round)
                    if is_orig
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        dx, dy = x2 - x1, y2 - y1
                        blen = sqrt(dx^2 + dy^2)
                        if blen > 1e-6
                            nx, ny = -dy / blen * 0.15, dx / blen * 0.15
                        else
                            nx, ny = 0.1, 0.0
                        end
                        text!(ax, mx + nx, my + ny;
                              text="$(round(real(eval); sigdigits=4))",
                              fontsize=9, align=(:center, :center), color=:gray30)
                    end
                end
            end
        end
    end
end

function _draw_lattice_bonds!(ax, lattice, all_coords, all_mdata,
                               e_dict, pattern, unique_sites, Ni, Nj, n_repeat,
                               e_min, e_max)
end

# ============================================================================
# Bond partner logic
# ============================================================================

"""
    _is_cross_diagonal(bond_type) → Bool

Detect the / cross-diagonal direction. Matches:
- "Diagonal/" or "/" (Square convention)
- "Diagonal2" (Honeycomb J1J2 convention)
Must be checked BEFORE generic "Diagonal" match.
"""
function _is_cross_diagonal(bond_type::String)
    occursin("/", bond_type) && return true
    occursin("Diagonal2", bond_type) && return true
    return false
end

"""
    _is_forward_diagonal(bond_type) → Bool

Detect the \\ forward-diagonal direction. Matches:
- "Diagonal\\" or "\\" (Square convention)
- "Diagonal1" (Honeycomb J1J2 convention)
- generic "Diagonal" (fallback)
"""
function _is_forward_diagonal(bond_type::String)
    occursin("\\", bond_type) && return true
    occursin("Diagonal1", bond_type) && return true
    occursin("Diagonal", bond_type) && return true   # generic fallback
    return false
end

"""
Bond offsets for honeycomb brickwall.
Returns ((di1,dj1), (di2,dj2)) as raw offsets from the anchor site (i,j).
"""
function _bond_offsets_honeycomb(bond_type::String)
    if occursin("Vertical", bond_type) || occursin("Jy", bond_type)
        return (0, 0), (1, 0)
    elseif occursin("Horizontal", bond_type) && occursin("J2", bond_type)
        return (0, 0), (0, 2)
    elseif occursin("Horizontal", bond_type) || occursin("Jx", bond_type) || occursin("Jz", bond_type)
        return (0, 0), (0, 1)
    elseif _is_cross_diagonal(bond_type)
        return (0, 1), (1, 0)    # / direction
    elseif _is_forward_diagonal(bond_type)
        return (0, 0), (1, 1)    # \ direction
    else
        return (0, 0), (0, 1)
    end
end

"""
Bond offsets for Square lattice. Same diagonal logic as Honeycomb.
"""
function _bond_offsets_square(bond_type::String)
    if occursin("Vertical", bond_type) || occursin("vertical", bond_type)
        return (0, 0), (1, 0)
    elseif occursin("Horizontal", bond_type) || occursin("horizontal", bond_type)
        return (0, 0), (0, 1)
    elseif _is_cross_diagonal(bond_type)
        return (0, 1), (1, 0)    # / direction
    elseif _is_forward_diagonal(bond_type)
        return (0, 0), (1, 1)    # \ direction
    else
        return (0, 0), (0, 1)
    end
end
