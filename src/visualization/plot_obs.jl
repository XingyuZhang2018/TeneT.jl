# Observable visualization using Makie.jl
# Provides convergence curve plotting and lattice observable visualization.

using CairoMakie

"""
    ObsPlotter

Accumulates observable history across iterations and generates plots.
"""
mutable struct ObsPlotter
    energies::Vector{Float64}
    mag_norms::Vector{Float64}
    cor_lens::Vector{Float64}
    χ_values::Vector{Int}
    save_path::String
    save_format::String
end

function ObsPlotter(save_path::String; save_format::String="png")
    isdir(save_path) || mkpath(save_path)
    return ObsPlotter(Float64[], Float64[], Float64[], Int[], save_path, save_format)
end

# ============================================================================
# Convergence plot
# ============================================================================

function plot_convergence!(plotter::ObsPlotter, e, mag, ξ, χ::Int)
    push!(plotter.energies, real(e[1]))
    push!(plotter.mag_norms, real(mag[1]))
    push!(plotter.cor_lens, real(ξ))
    push!(plotter.χ_values, χ)

    n = length(plotter.energies)
    steps = collect(1:n)

    fig = Figure(size=(700, 900), fontsize=14)

    function _make_panel(pos, ylabel, data, color; title="")
        ax = Axis(fig[pos, 1]; xlabel=(pos == 3 ? "Step" : ""),
                  ylabel=ylabel, title=title,
                  xticks=LinearTicks(min(n, 10)),
                  xgridvisible=true, ygridvisible=true,
                  xgridstyle=:dash, ygridstyle=:dash,
                  xgridcolor=(:black, 0.1), ygridcolor=(:black, 0.1))
        if n == 1
            scatter!(ax, steps, data; color=color, markersize=10)
            yval = data[1]
            margin = max(abs(yval) * 0.1, 1e-6)
            ylims!(ax, yval - margin, yval + margin)
            xlims!(ax, 0.5, 1.5)
        else
            lines!(ax, steps, data; color=color, linewidth=2)
            scatter!(ax, steps, data; color=color, markersize=8)
            xlims!(ax, 0.5, n + 0.5)
        end
        return ax
    end

    ax1 = _make_panel(1, "Energy / site", plotter.energies, :steelblue; title="Observable Convergence")
    ax2 = _make_panel(2, "|M| mean", plotter.mag_norms, :crimson)
    ax3 = _make_panel(3, "ξ", plotter.cor_lens, :seagreen)

    for (i, χv) in enumerate(plotter.χ_values)
        if i == 1 || χv != plotter.χ_values[i-1]
            text!(ax1, i + 0.05, plotter.energies[i];
                  text="χ=$χv", fontsize=11, color=:gray40,
                  align=(:left, :bottom))
        end
    end

    outfile = joinpath(plotter.save_path, "convergence.$(plotter.save_format)")
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

    # pattern_value → first (i,j) position stored in m_dict
    unique_sites = Dict{Int, Tuple{Int,Int}}()
    for (key, _) in m_dict
        parts = split(key, ",")
        i, j = parse(Int, parts[1]), parse(Int, parts[2])
        unique_sites[pattern[i, j]] = (i, j)
    end

    # Build tiled grid
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

    # Collect all bond energy magnitudes for linewidth scaling
    all_evals = Float64[]
    for (_, bond_data) in e_dict
        for (_, ev) in bond_data
            push!(all_evals, abs(real(ev)))
        end
    end
    e_max = isempty(all_evals) ? 1.0 : maximum(all_evals)
    e_min = isempty(all_evals) ? 0.0 : minimum(all_evals)

    # Figure size: make it roughly square, favoring width
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

    # --- Draw bonds (behind sites) with width ~ |energy| ---
    _draw_lattice_bonds!(ax, lattice_type, all_coords, all_mdata,
                         e_dict, pattern, unique_sites, Ni, Nj, n_repeat,
                         e_min, e_max)

    # --- Draw sites as gray circles ---
    for (k, (x, y)) in all_coords
        alpha = is_original[k] ? 1.0 : 0.4
        ms = is_original[k] ? 22 : 16
        scatter!(ax, [x], [y]; color=(:gray70, alpha), markersize=ms,
                 strokewidth=is_original[k] ? 1.5 : 0.5,
                 strokecolor=(:gray40, alpha))
    end

    # --- Draw magnetization arrows (original unit cell only) ---
    # Arrow convention: Mx → horizontal, Mz → vertical (up = +Mz)
    # Arrow length scaled so max |M| ≈ 0.45 (fits within site spacing)
    mag_max_val = maximum(abs(real(all_mdata[k]["|M|"])) for k in keys(all_coords))
    arrow_scale = mag_max_val > 1e-10 ? 0.45 / mag_max_val : 0.0
    for (k, (x, y)) in all_coords
        is_original[k] || continue
        mdata = all_mdata[k]
        mx = real(mdata["Mx"]) * arrow_scale
        mz = real(mdata["Mz"]) * arrow_scale
        amag = sqrt(mx^2 + mz^2)
        amag < 1e-8 && continue
        arrows!(ax, [x], [y], [mx], [mz];
                color=(:black, 0.9), linewidth=2.5,
                arrowsize=12, arrowcolor=(:black, 0.9))
    end

    # --- Label original unit-cell sites with index ---
    for (k, (x, y)) in all_coords
        is_original[k] || continue
        mdata = all_mdata[k]
        ci, cj = mod1(k[1], Ni), mod1(k[2], Nj)
        text!(ax, x, y + 0.55; text="($ci,$cj)", fontsize=10, color=:gray20,
              align=(:center, :bottom))
    end

    # Padding
    xmargin = max(1.5, xspan * 0.12)
    ymargin = max(1.5, yspan * 0.12)
    xlims!(ax, minimum(all_xs) - xmargin, maximum(all_xs) + xmargin)
    ylims!(ax, minimum(all_ys) - ymargin, maximum(all_ys) + ymargin)

    # Legend for bond types
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

"""
Honeycomb brickwall → true hexagonal coordinates.
B sublattice ((i+j) odd) shifted down by 0.5.
All bonds length=1, angles 120° apart: Jy vertical, Jx/Jz diagonal ±30°.
"""
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

# Bond colors matching Kitaev convention: Jx=cyan, Jy=pink/salmon, Jz=green
const _BOND_COLORS = Dict(
    "Jx" => colorant"#00BFFF",    # cyan
    "Jy" => colorant"#FF8080",    # salmon/pink
    "Jz" => colorant"#80FF80",    # light green
    "J1_Horizontal" => :royalblue, "J1_Vertical" => :forestgreen,
    "J2_Horizontal" => :orange, "J2_Diagonal1" => :purple, "J2_Diagonal2" => :hotpink,
)

function _bond_color(bond_type::String)
    for (key, col) in _BOND_COLORS
        occursin(key, bond_type) && return col
    end
    return :gray60
end

"""
Compute bond linewidth from energy value. Linewidth scales linearly with |energy|.
Min width = 2, max width = 12.
"""
function _bond_linewidth(eval, e_min, e_max)
    ae = abs(real(eval))
    if e_max ≈ e_min
        return 14.0
    end
    t = (ae - e_min) / (e_max - e_min)  # 0 to 1
    return 5.0 + t * 18.0               # 5 to 23
end

function _draw_lattice_bonds!(ax, ::Honeycomb{:brickwall}, all_coords, all_mdata,
                               e_dict, pattern, unique_sites, Ni, Nj, n_repeat,
                               e_min, e_max)
    # Build reverse map: pattern_value → all (ci,cj) positions in unit cell
    pval_positions = Dict{Int, Vector{Tuple{Int,Int}}}()
    for ci in 1:Ni, cj in 1:Nj
        pv = pattern[ci, cj]
        push!(get!(pval_positions, pv, Tuple{Int,Int}[]), (ci, cj))
    end

    for (bond_type, bond_data) in e_dict
        color = _bond_color(bond_type)
        for (pos_str, eval) in bond_data
            parts = split(pos_str, ",")
            oi, oj = parse(Int, parts[1]), parse(Int, parts[2])
            pv = pattern[oi, oj]
            lw = _bond_linewidth(eval, e_min, e_max)

            # Draw this bond from ALL unit-cell positions with the same pattern value
            for (ci, cj) in pval_positions[pv]
                pi2, pj2 = _bond_partner_honeycomb(bond_type, ci, cj, Ni, Nj)

                for di in 0:(n_repeat-1), dj in 0:(n_repeat-1)
                    gi1 = ci + di * Ni
                    gj1 = cj + dj * Nj
                    gi2 = pi2 + di * Ni
                    gj2 = pj2 + dj * Nj
                    if pi2 < ci; gi2 = pi2 + (di + 1) * Ni; end
                    if pj2 < cj; gj2 = pj2 + (dj + 1) * Nj; end

                    haskey(all_coords, (gi1, gj1)) || continue
                    haskey(all_coords, (gi2, gj2)) || continue

                    x1, y1 = all_coords[(gi1, gj1)]
                    x2, y2 = all_coords[(gi2, gj2)]

                    is_orig = (di == 0 && dj == 0)
                    alpha = is_orig ? 0.85 : 0.3
                    lines!(ax, [x1, x2], [y1, y2]; color=(color, alpha),
                           linewidth=lw, linecap=:round)

                    # Energy label on original unit-cell bonds
                    if is_orig
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        elabel = "$(round(real(eval); sigdigits=4))"
                        # Offset label perpendicular to bond direction
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
        for (pos_str, eval) in bond_data
            parts = split(pos_str, ",")
            oi, oj = parse(Int, parts[1]), parse(Int, parts[2])
            pv = pattern[oi, oj]
            lw = _bond_linewidth(eval, e_min, e_max)

            for (ci, cj) in pval_positions[pv]
                pi2, pj2 = _bond_partner_square(bond_type, ci, cj, Ni, Nj)
                for di in 0:(n_repeat-1), dj in 0:(n_repeat-1)
                    gi1 = ci + di * Ni
                    gj1 = cj + dj * Nj
                    gi2 = pi2 + di * Ni
                    gj2 = pj2 + dj * Nj
                    if pi2 < ci; gi2 = pi2 + (di + 1) * Ni; end
                    if pj2 < cj; gj2 = pj2 + (dj + 1) * Nj; end

                    haskey(all_coords, (gi1, gj1)) || continue
                    haskey(all_coords, (gi2, gj2)) || continue

                    x1, y1 = all_coords[(gi1, gj1)]
                    x2, y2 = all_coords[(gi2, gj2)]
                    is_orig = (di == 0 && dj == 0)
                    alpha = is_orig ? 0.85 : 0.3
                    lines!(ax, [x1, x2], [y1, y2]; color=(color, alpha),
                           linewidth=lw, linecap=:round)
                    if is_orig && ci == oi && cj == oj
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        text!(ax, mx + 0.08, my + 0.08;
                              text="$(round(real(eval); sigdigits=4))",
                              fontsize=8, align=(:left, :bottom), color=(color, 0.9))
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

function _bond_partner_honeycomb(bond_type::String, i, j, Ni, Nj)
    if occursin("Vertical", bond_type) || occursin("Jy", bond_type)
        return (mod1(i + 1, Ni), j)
    elseif occursin("Horizontal", bond_type) && occursin("J2", bond_type)
        return (i, mod1(j + 2, Nj))
    elseif occursin("Horizontal", bond_type) || occursin("Jx", bond_type) || occursin("Jz", bond_type)
        return (i, mod1(j + 1, Nj))
    elseif occursin("Diagonal1", bond_type)
        return (mod1(i + 1, Ni), mod1(j + 1, Nj))
    elseif occursin("Diagonal2", bond_type)
        return (mod1(i + 1, Ni), mod1(j + 1, Nj))
    else
        return (i, mod1(j + 1, Nj))
    end
end

function _bond_partner_square(bond_type::String, i, j, Ni, Nj)
    if occursin("Vertical", bond_type) || occursin("vertical", bond_type)
        return (mod1(i + 1, Ni), j)
    elseif occursin("Horizontal", bond_type) || occursin("horizontal", bond_type)
        return (i, mod1(j + 1, Nj))
    elseif occursin("Diagonal", bond_type) || occursin("diagonal", bond_type)
        return (mod1(i + 1, Ni), mod1(j + 1, Nj))
    else
        return (i, mod1(j + 1, Nj))
    end
end

# ============================================================================
# Combined callback
# ============================================================================

function plot_observable_callback!(plotter::ObsPlotter, e, mag, ξ, χ::Int,
                                   lattice_type, pattern::Matrix{Int})
    plot_convergence!(plotter, e, mag, ξ, χ)
    plot_lattice_obs(e[2], mag[2], lattice_type, pattern;
                     save_path=plotter.save_path, save_format=plotter.save_format, χ=χ)
    return nothing
end
