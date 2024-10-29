using ForwardDiffChainRules
using ChainRulesCore

function my_function(x, y)
    return x * y + sin(x)
end

function ChainRulesCore.frule((Δx, Δy), ::typeof(my_function), x, y)
    # Primal computation
    primal = my_function(x, y)
    
    # Tangent computation
    ∂f_∂x = y + cos(x)
    ∂f_∂y = x
    tangent = ∂f_∂x * Δx + ∂f_∂y * Δy
    
    return primal, tangent
end

x, y = 3.0, 4.0
Δx, Δy = 1.0, 0.0  # Unit tangent in the x direction
primal, tangent = ChainRulesCore.frule((Δx, Δy), my_function, x, y)

println("Primal: $primal")
println("Tangent: $tangent")

# Validate using ForwardDiff
diff_result = ForwardDiff.derivative(t -> my_function(t, y), x)
println("ForwardDiff result: $diff_result")
# # create a ForwardDiff-dispatch for scalar type `x1` and `x2`
# @ForwardDiff_frule f1(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual)

# # create a ForwardDiff-dispatch for vector type `x1` and `x2`
# @ForwardDiff_frule f1(x1::AbstractVector{<:ForwardDiff.Dual}, x2::AbstractVector{<:ForwardDiff.Dual})

# # create a ForwardDiff-dispatch for matrix type `x1` and `x2`
# @ForwardDiff_frule f1(x1::AbstractMatrix{<:ForwardDiff.Dual}, x2::AbstractMatrix{<:ForwardDiff.Dual})