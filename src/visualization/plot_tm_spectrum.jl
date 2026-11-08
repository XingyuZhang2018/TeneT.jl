# Transfer-matrix spectrum visualization.

function _tm_momentum_from_filename(filename::AbstractString)
    matched = match(r"^k(.+)\.log$", filename)
    matched === nothing && return nothing

    value = matched.captures[1]
    rational = match(r"^([+-]?\d+)_over_([+-]?\d+)$", value)
    if rational !== nothing
        numerator_value = parse(Int, rational.captures[1])
        denominator_value = parse(Int, rational.captures[2])
        iszero(denominator_value) && return nothing
        return numerator_value / denominator_value
    end
    return tryparse(Float64, value)
end

"""
    _read_TM_spectrum(folder; nlevels=nothing)

Read finite transfer-matrix gaps from `k*.log` files. When `nlevels` is not
given, the largest level count found in the directory is used. Files with a
different number of levels are treated as incomplete and skipped.
"""
function _read_TM_spectrum(folder::AbstractString;
                           nlevels::Union{Nothing, Int}=nothing)
    isdir(folder) ||
        throw(ArgumentError("TM spectrum folder does not exist: $folder"))
    isnothing(nlevels) || nlevels > 0 ||
        throw(ArgumentError("nlevels must be positive; got $nlevels."))

    parsed = Dict{Float64, Vector{Float64}}()
    for filename in readdir(folder)
        k = _tm_momentum_from_filename(filename)
        k === nothing && continue

        lines = filter(!isempty, strip.(readlines(joinpath(folder, filename))))
        isempty(lines) && continue
        values = try
            parse.(Float64, lines)
        catch error
            error isa ArgumentError || rethrow()
            continue
        end
        all(isfinite, values) || continue
        parsed[Float64(k)] = values
    end

    isempty(parsed) &&
        throw(ArgumentError("No finite TM spectrum logs found in $folder."))
    expected_levels = isnothing(nlevels) ?
                      maximum(length, values(parsed)) : nlevels
    complete = Dict(
        k => parsed[k] for k in sort(collect(keys(parsed)))
        if length(parsed[k]) == expected_levels
    )
    isempty(complete) &&
        throw(ArgumentError("No TM spectrum logs in $folder contain $expected_levels levels."))
    return complete
end

"""
    plot_TM_spectrum(results; kwargs...)

Plot transfer-matrix gaps stored as `momentum => gaps`. Dictionary keys are
used directly as the horizontal coordinate.
"""
function plot_TM_spectrum(results::AbstractDict;
                          save_path::AbstractString="spectrum.png",
                          title::AbstractString="Transfer-matrix spectrum",
                          xlabel::AbstractString="k / pi",
                          ylabel::AbstractString="gap",
                          xlimits=nothing,
                          size=(960, 640),
                          markersize::Real=6,
                          linewidth::Real=1.5)
    isempty(results) &&
        throw(ArgumentError("Cannot plot an empty TM spectrum."))

    ordered_k = sort(collect(keys(results)))
    all(k -> k isa Real && isfinite(k), ordered_k) ||
        throw(ArgumentError("TM spectrum momenta must be finite real numbers."))

    level_counts = unique(length(results[k]) for k in ordered_k)
    length(level_counts) == 1 ||
        throw(ArgumentError("All TM spectrum points must contain the same number of levels."))
    nlevels = only(level_counts)
    nlevels > 0 ||
        throw(ArgumentError("TM spectrum points must contain at least one level."))
    all(k -> all(isfinite, results[k]), ordered_k) ||
        throw(ArgumentError("TM spectrum gaps must be finite."))

    fig = Figure(size=size)
    ax = Axis(fig[1, 1]; xlabel, ylabel, title)
    for band in 1:nlevels
        band_values = [results[k][band] for k in ordered_k]
        scatterlines!(
            ax,
            ordered_k,
            band_values;
            markersize,
            linewidth,
        )
    end
    xlimits === nothing || xlims!(ax, xlimits...)

    mkpath(dirname(abspath(save_path)))
    save(save_path, fig)
    return fig
end

"""
    plot_TM_spectrum(folder; nlevels=nothing, xscale=1, kwargs...)

Read `k*.log` files from `folder` and plot them. Logged momenta are divided by
`xscale`, so use `xscale=pi` to display physical momentum as `k / pi`.
"""
function plot_TM_spectrum(folder::AbstractString;
                          nlevels::Union{Nothing, Int}=nothing,
                          xscale::Real=1,
                          save_path::AbstractString=joinpath(folder, "spectrum.png"),
                          kwargs...)
    isfinite(xscale) && !iszero(xscale) ||
        throw(ArgumentError("xscale must be finite and nonzero; got $xscale."))
    results = _read_TM_spectrum(folder; nlevels)
    scaled_results = Dict(k / xscale => gaps for (k, gaps) in results)
    return plot_TM_spectrum(scaled_results; save_path, kwargs...)
end
