# Symmetry restrictions, central canonical forms, gauge transforms, and MCF for iPEPS tensors.

"""
    C4v_restriction(A)

Impose C4v symmetry on an iPEPS tensor `A` with indices `(l, d, r, u, p, n)` (6-leg)
by averaging over all C4v symmetry operations (reflections and rotations).
"""
function C4v_restriction(A::AbstractArray{<:Number, 6})
    A += permutedims(conj(A), (1,4,3,2,5,6)) # up-down reflection
    A += permutedims(conj(A), (3,2,1,4,5,6)) # left-right reflection
    A += permutedims(conj(A), (2,1,4,3,5,6)) # diagonal reflection
    A += permutedims(conj(A), (4,3,2,1,5,6)) # rotation
    return A
end

"""
    C4v_restriction(A)

Impose C4v symmetry on an iPEPS tensor `A` with indices `(l, d, r, u, p)` (5-leg)
by averaging over all C4v symmetry operations (reflections and rotations).
"""
function C4v_restriction(A::AbstractArray{<:Number, 5})
    A += permutedims(conj(A), (1,4,3,2,5)) # up-down reflection
    A += permutedims(conj(A), (3,2,1,4,5)) # left-right reflection
    A += permutedims(conj(A), (2,1,4,3,5)) # diagonal reflection
    A += permutedims(conj(A), (4,3,2,1,5)) # rotation
    return A
end

"""
    _restriction_ipeps(A)

Default restriction for iPEPS tensor. Returns `A` unchanged (identity restriction).
Override for specific symmetry constraints.
```
        4
        |
 1 -- ipeps -- 3
        |
        2
```
"""
function _restriction_ipeps(A)
    return A
end

# --- Central canonical forms ---

"""
    central_canonical1(A)

Transform iPEPS tensor `A` into central canonical form (variant 1).
Uses `pepsgeneral` to compute the canonical core and `ARstoA1` to reconstruct.
"""
function central_canonical1(A)
    Ac, Rs = pepsgeneral(A)
    A = ARstoA1(Ac, Rs)
    return A
end

"""
    central_canonical2(A)

Transform iPEPS tensor `A` into central canonical form (variant 2).
Uses `pepsgeneral` to compute the canonical core and `ARstoA2` to reconstruct.
"""
function central_canonical2(A)
    Ac, Rs = pepsgeneral(A)
    A = ARstoA2(Ac, Rs)
    return A
end

"""
    pepsgeneral(A; tol=1e-12)

Compute the central canonical form of a 6-leg iPEPS tensor `A` (indices: l, d, r, u, p, n).
Returns `(Ac, Rs)` where `Ac` is the canonical core tensor and `Rs` is a vector of 4
upper-triangular matrices (one per virtual leg).

Iteratively applies QR decompositions and rotations until convergence.
"""
function pepsgeneral(A::AbstractArray{<:Number, 6}; tol=1e-12)
    function leftorth(A)
        D, d, N = size(A)[[1,5,6]]
        Q, R = qrpos(reshape(permutedims(A, (1, 2, 3, 5, 4, 6)), D^3 * d, D))
        Q = permutedims(reshape(Q, D, D, D, d, D, N), (1, 2, 3, 5, 4, 6))
        return Q, R
    end

    function rotate(A)
        return permutedims(A, (2, 3, 4, 1, 5, 6))
    end

    D = size(A, 1)
    Rs = Zygote.Buffer([_arraytype(A)(rand(eltype(A), D, D)) for _ in 1:4])
    for i in 1:4
        A, R = leftorth(A)
        A = rotate(A)
        Rs[i] = R
    end
    conv = Inf
    iter = 0
    while conv > tol
        conv_sum = Zygote.@ignore 0
        for i in 1:4
            A, R = leftorth(A)
            A = rotate(A)
            Rs[i] = R * Rs[i]
            conv_sum += norm(R - one(R))
        end
        conv = conv_sum
        iter += 1
        if iter > 100
            @warn "pepsgeneral did not converge within 100 iterations, conv=$conv"
            break
        end
    end
    return A, copy(Rs)
end

"""
    pepsgeneral(A; tol=1e-12)

Compute the central canonical form of a 5-leg iPEPS tensor `A` (indices: l, d, r, u, p).
Returns `(Ac, Rs)` where `Ac` is the canonical core tensor and `Rs` is a vector of 4
upper-triangular matrices (one per virtual leg).
"""
function pepsgeneral(A::AbstractArray{<:Number, 5}; tol=1e-12)
    function leftorth(A)
        D, d = size(A)[[1,5]]
        Q, R = qrpos(reshape(permutedims(A, (1, 2, 3, 5, 4)), D^3 * d, D))
        Q = permutedims(reshape(Q, D, D, D, d, D), (1, 2, 3, 5, 4))
        return Q, R
    end

    function rotate(A)
        return permutedims(A, (2, 3, 4, 1, 5))
    end

    D = size(A, 1)
    Rs = Zygote.Buffer([rand(eltype(A), D, D) for _ in 1:4])
    for i in 1:4
        A, R = leftorth(A)
        A = rotate(A)
        Rs[i] = R
    end
    conv = Inf
    iter = 0
    while conv > tol
        conv_sum = Zygote.@ignore 0
        for i in 1:4
            A, R = leftorth(A)
            A = rotate(A)
            Rs[i] = R * Rs[i]
            conv_sum += norm(R - one(R))
        end
        conv = conv_sum
        iter += 1
        if iter > 100
            @warn "pepsgeneral did not converge within 100 iterations, conv=$conv"
            break
        end
    end
    return A, copy(Rs)
end

"""
    ARstoA1(A, Rs)

Reconstruct iPEPS tensor from canonical core `A` and R-matrices `Rs` (variant 1).
Balances the gauge by SVD decomposition of `R[3]*R[1]'` and `R[4]*R[2]'`.
6-leg version (l, d, r, u, p, n).
"""
function ARstoA1(A::AbstractArray{<:Number, 6}, Rs)
    @tensor R31[a,c] := Rs[3][a,b] * Rs[1][c,b]
    F = svd(R31)
    sqrtS = diagm(sqrt.(F.S))
    Rs3 = F.U * sqrtS
    Rs1 = conj(F.V * sqrtS)

    @tensor R42[a,c] := Rs[4][a,b] * Rs[2][c,b]
    F = svd(R42)
    sqrtS = diagm(sqrt.(F.S))
    Rs4 = F.U * sqrtS
    Rs2 = conj(F.V * sqrtS)

    return @tensor Aout[e,f,g,h,p,n] := A[a,b,c,d,p,n] * Rs1[d,h] * Rs2[a,e] * Rs3[b,f] * Rs4[c,g]
end

"""
    ARstoA2(A, Rs)

Reconstruct iPEPS tensor from canonical core `A` and R-matrices `Rs` (variant 2).
Uses simpler gauge construction with `R[3]*R[1]'` and `R[4]*R[2]'` directly.
6-leg version (l, d, r, u, p, n).
"""
function ARstoA2(A::AbstractArray{<:Number, 6}, Rs)
    @tensor R3[a,c] := Rs[3][a,b] * Rs[1][c,b]
    @tensor R4[a,c] := Rs[4][a,b] * Rs[2][c,b]

    return @tensor Aout[a,f,g,d,p,n] := A[a,b,c,d,p,n] * R3[b,f] * R4[c,g]
end

"""
    ARstoA(A, Rs)

Reconstruct iPEPS tensor from canonical core `A` and R-matrices `Rs`.
5-leg version (l, d, r, u, p). Balances the gauge via SVD.
"""
function ARstoA(A::AbstractArray{<:Number, 5}, Rs)
    R31 = Rs[3] * transpose(Rs[1])
    F = svd(R31)
    sqrtS = Diagonal(sqrt.(F.S))
    Rs3 = F.U * sqrtS
    Rs1 = F.V * sqrtS

    R42 = Rs[4] * transpose(Rs[2])
    F = svd(R42)
    sqrtS = Diagonal(sqrt.(F.S))
    Rs4 = F.U * sqrtS
    Rs2 = F.V * sqrtS

    return @tensor out[e,f,g,h,p] := A[a,b,c,d,p] * Rs1[d,h] * Rs2[a,e] * Rs3[b,f] * Rs4[c,g]
end

"""
    pepsgeneral_Ac(A; tol=1e-12)

Compute only the canonical core tensor `Ac` from iPEPS tensor `A` (6-leg version),
discarding the R-matrices. Useful when only the isometric core is needed.
"""
function pepsgeneral_Ac(A::AbstractArray{<:Number, 6}; tol=1e-12)
    function leftorth(A)
        D, d, N = size(A)[[1,5,6]]
        Q, R = qrpos(reshape(permutedims(A, (1, 2, 3, 5, 4, 6)), D^3 * d, D))
        Q = permutedims(reshape(Q, D, D, D, d, D, N), (1, 2, 3, 5, 4, 6))
        return Q, R
    end

    function rotate(A)
        return permutedims(A, (2, 3, 4, 1, 5, 6))
    end

    D = size(A, 1)
    for i in 1:4
        A, R = leftorth(A)
        A = rotate(A)
    end
    conv = Inf
    iter = 0
    while conv > tol
        conv_sum = Zygote.@ignore 0
        for i in 1:4
            A, R = leftorth(A)
            A = rotate(A)
            conv_sum += norm(R - one(R))
        end
        conv = conv_sum
        iter += 1
        if iter > 100
            @warn "pepsgeneral did not converge within 100 iterations, conv=$conv"
            break
        end
    end
    return A
end

"""
    pepsgeneral_Ac(A; tol=1e-12)

Compute only the canonical core tensor `Ac` from iPEPS tensor `A` (5-leg version),
discarding the R-matrices.
"""
function pepsgeneral_Ac(A::AbstractArray{<:Number, 5}; tol=1e-12)
    function leftorth(A)
        D, d = size(A)[[1,5]]
        Q, R = qrpos(reshape(permutedims(A, (1, 2, 3, 5, 4)), D^3 * d, D))
        Q = permutedims(reshape(Q, D, D, D, d, D), (1, 2, 3, 5, 4))
        return Q, R
    end

    function rotate(A)
        return permutedims(A, (2, 3, 4, 1, 5))
    end

    D = size(A, 1)
    for i in 1:4
        A, _ = leftorth(A)
        A = rotate(A)
    end
    conv = Inf
    iter = 0
    while conv > tol
        conv_sum = Zygote.@ignore 0
        for i in 1:4
            A, R = leftorth(A)
            A = rotate(A)
            conv_sum += norm(R - one(R))
        end
        conv = conv_sum
        iter += 1
        if iter > 100
            @warn "pepsgeneral did not converge within 100 iterations, conv=$conv"
            break
        end
    end
    return A
end

# --- Local gauge operations ---

"""
    local_gauge_contraction(A, G)

Apply local gauge matrices `G = [G1, G2, G3, G4]` to a 5-leg iPEPS tensor `A`.
Returns `G1 * A * G2 * G3 * G4` contracted on the four virtual legs.
"""
local_gauge_contraction(A, G) = @tensor out[e,f,g,h,p] := A[a,b,c,d,p] * G[1][e,a] * G[2][b,f] * G[3][c,g] * G[4][h,d]

"""
    guage_transfer(A, G, params)

Apply gauge transformation to all sites of a multi-site iPEPS tensor `A` (6-leg, last index = site).
`G = [Gh, Gv]` are horizontal and vertical gauge matrices indexed by site number.
Uses `params.pattern` to determine the unit cell layout.
"""
function guage_transfer(A, G, params)
    Gh, Gv = G
    pattern = params.pattern
    Ni, Nj = size(pattern)
    A_buf = Zygote.Buffer(A)
    for q in 1:size(A, 6)
        i, j = Tuple(findfirst(==(q), pattern))
        ir = mod1(i - 1, Ni)
        jr = mod1(j - 1, Nj)
        A_buf[:,:,:,:,:,q] = local_gauge_contraction(
            A[:,:,:,:,:,q],
            [inv(Gh[:,:,pattern[i,jr]]), Gv[:,:,q], Gh[:,:,q], inv(Gv[:,:,pattern[ir,j]])]
        )
    end
    return copy(A_buf)
end

"""
    find_local_min_norm_G(A, params)

Find gauge matrices `G = [Gh, Gv]` that minimize the Frobenius norm of the
gauge-transformed iPEPS tensor. Uses LBFGS optimization from OptimKit.
"""
function find_local_min_norm_G(A, params)
    atype = _arraytype(A)
    A_cpu = Array(A)

    function f(G)
        A_prime = guage_transfer(A_cpu, G, params)
        return norm(A_prime)
    end

    function fg(G)
        cost, vjp = Zygote.pullback(f, G)
        g = vjp(1)[1]
        return cost, g
    end

    D, N = size(A)[[1, 6]]
    eltypeA = eltype(A)
    Gh = randn(eltypeA, D, D, N)
    Gv = randn(eltypeA, D, D, N)
    for q in 1:N
        Gh[:,:,q] = I(D)
        Gv[:,:,q] = I(D)
    end
    Ginit = [Gh, Gv]
    @info "initial norm = $(f(Ginit))"

    G, fval, _ = optimize(fg, Ginit, LBFGS(maxiter=1000, gradtol=1e-15))
    @info "final norm = $fval"

    return atype.(G)
end

"""
    local_min_norm(A, params; ifignore_gauge=true)

Apply the minimum-norm gauge transformation to iPEPS tensor `A`.
If `ifignore_gauge=true`, the gauge optimization is excluded from AD.
"""
function local_min_norm(A, params; ifignore_gauge=true)
    if ifignore_gauge
        G = Zygote.@ignore find_local_min_norm_G(A, params)
    else
        G = find_local_min_norm_G(A, params)
    end
    AG = guage_transfer(A, G, params)
    return AG
end

"""
    ChainRulesCore.rrule(::typeof(find_local_min_norm_G), A, params)

Custom reverse-mode AD rule for `find_local_min_norm_G`.
Uses implicit differentiation through the gauge optimality condition
via a linear solve to propagate gradients.
"""
function ChainRulesCore.rrule(::typeof(find_local_min_norm_G), A, params)
    G = find_local_min_norm_G(A, params)
    atype = _arraytype(A)
    function find_local_min_norm_G_pullback(DeltaG)
        DeltaG = Array.(DeltaG)
        A_arr = Array(A)
        G_arr = Array.(G)
        function fixpoint(A_in, G_in)
            AG = guage_transfer(A_in, G_in, params)

            @tensor Ml[1,6] := AG[1,2,3,4,5,7] * conj(AG[6,2,3,4,5,7])
            @tensor Mr[6,3] := AG[1,2,3,4,5,7] * conj(AG[1,2,6,4,5,7])

            @tensor Mu[4,6] := AG[1,2,3,4,5,7] * conj(AG[1,2,3,6,5,7])
            @tensor Md[6,2] := AG[1,2,3,4,5,7] * conj(AG[1,6,3,4,5,7])

            return [Ml - Mr, Mu - Md]
        end

        cost, vjp = Zygote.pullback(fixpoint, A_arr, G_arr)
        sum(norm.(cost)) >= 1e-10 && @warn "local min norm gauge condition not satisfied, cost=$cost"
        vjp_A(x) = vjp(x)[1]
        vjp_G(x) = vjp(x)[2]

        dA, info = linsolve(vjp_G, -DeltaG; maxiter=1)
        if info == 0
            @warn "linsolve did not converge in find_local_min_norm_G_pullback, info=$info"
        end
        @show norm(dA)

        return NoTangent(), atype(vjp_A(dA)), NoTangent()
    end
    return G, find_local_min_norm_G_pullback
end

"""
    find_local_hermite_G(A, params)

Find gauge matrices `G = [Gh, Gv]` that make the iPEPS tensor as close to
Hermitian-symmetric as possible (left-right and up-down reflection symmetry).
Uses LBFGS optimization from OptimKit.
"""
function find_local_hermite_G(A, params)
    atype = _arraytype(A)
    A_cpu = Zygote.@ignore Array(A)

    function f(G)
        A_prime = guage_transfer(A_cpu, G, params)
        return norm(A_prime - permutedims(A_prime, (3,2,1,4,5,6))) + norm(A_prime - permutedims(A_prime, (1,4,3,2,5,6)))
    end

    function fg(G)
        cost, vjp = Zygote.pullback(f, G)
        g = vjp(1)[1]
        return cost, g
    end

    D, N = size(A)[[1, 6]]
    eltypeA = eltype(A)
    Gh = randn(eltypeA, D, D, N)
    Gv = randn(eltypeA, D, D, N)
    for q in 1:N
        Gh[:,:,q] = I(D)
        Gv[:,:,q] = I(D)
    end
    Ginit = [Gh, Gv]
    @info "MCF initial norm = $(f(Ginit))"

    G, fval, _ = optimize(fg, Ginit, LBFGS(maxiter=100))
    @info "MCF final norm = $fval"

    return atype.(G)
end

"""
    local_hermite(A, params)

Apply the Hermite-symmetrizing gauge transformation to iPEPS tensor `A`.
The gauge optimization is excluded from AD (via Zygote.@ignore).
"""
function local_hermite(A, params)
    G = Zygote.@ignore find_local_hermite_G(A, params)
    AG = guage_transfer(A, G, params)
    return AG
end

# --- Minimal Canonical Form (MCF) ---

"""
    to_mcf_ipeps(T; max_iter=1000, tol=1e-12)

Transform a 5-leg iPEPS tensor `T` into Minimal Canonical Form (MCF).
Input tensor index order: (l, d, r, u, p).

MCF satisfies the condition that the reduced density matrices on opposite
virtual legs are transposes of each other, achieved by iteratively applying
balancing gauge transformations.
"""
function to_mcf_ipeps(T; max_iter=1000, tol=1e-12)
    Dl, Dd, Dr, Du, dp = size(T)
    @assert Dl == Dr "Horizontal virtual dimensions must be equal for MCF."
    @assert Dd == Du "Vertical virtual dimensions must be equal for MCF."

    curr_T = copy(T)
    max_diff = 0.0

    for iter in 1:max_iter
        # --- Horizontal direction (l & r) ---
        ML = reshape(curr_T, Dl, :)
        rho_l = ML * ML'

        MR = reshape(permutedims(curr_T, (3, 1, 2, 4, 5)), Dr, :)
        rho_r = MR * MR'

        target_r = transpose(rho_r)
        diff_h = norm(rho_l - target_r) / norm(rho_l)

        g_h = solve_balancing_gauge(rho_l, target_r)
        inv_gh_t = transpose(inv(g_h))

        # Update l (index 1)
        curr_T = reshape(g_h * ML, Dl, Dd, Dr, Du, dp)

        # Update r (index 3)
        tmp_r = reshape(permutedims(curr_T, (3, 1, 2, 4, 5)), Dr, :)
        curr_T = permutedims(reshape(inv_gh_t * tmp_r, Dr, Dl, Dd, Du, dp), (2, 3, 1, 4, 5))

        # --- Vertical direction (d & u) ---
        MD = reshape(permutedims(curr_T, (2, 1, 3, 4, 5)), Dd, :)
        rho_d = MD * MD'

        MU = reshape(permutedims(curr_T, (4, 1, 2, 3, 5)), Du, :)
        rho_u = MU * MU'

        target_u = transpose(rho_u)
        diff_v = norm(rho_d - target_u) / norm(rho_d)

        g_v = solve_balancing_gauge(rho_d, target_u)
        inv_gv_t = transpose(inv(g_v))

        # Update d (index 2)
        tmp_d = reshape(permutedims(curr_T, (2, 1, 3, 4, 5)), Dd, :)
        curr_T = permutedims(reshape(g_v * tmp_d, Dd, Dl, Dr, Du, dp), (2, 1, 3, 4, 5))

        # Update u (index 4)
        tmp_u = reshape(permutedims(curr_T, (4, 1, 2, 3, 5)), Du, :)
        curr_T = permutedims(reshape(inv_gv_t * tmp_u, Du, Dl, Dd, Dr, dp), (2, 3, 4, 1, 5))

        max_diff = max(diff_h, diff_v)
        if max_diff < tol
            @info "MCF converged in $iter iterations."
            return curr_T
        end
    end

    @warn "MCF did not fully converge. Final diff: $max_diff"
    return curr_T
end

"""
    solve_balancing_gauge(A, B; reg=1e-15)

Solve for the balancing gauge matrix `g` such that `g * A * g = B`.
Returns `g = sqrt(H)` where `H = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}`.
This minimizes the Frobenius norm of the transformed tensor.
"""
function solve_balancing_gauge(A, B; reg=1e-15)
    A_safe = A + reg * I
    B_safe = B + reg * I

    sqrtA = sqrt(Hermitian(A_safe))
    isqrtA = inv(sqrtA)
    H = isqrtA * sqrt(Hermitian(sqrtA * B_safe * sqrtA)) * isqrtA
    return sqrt(Hermitian(H))
end

"""
    local_min_norm_iter(A, params)

Apply MCF-based minimum norm transformation to a single-site iPEPS tensor.
Extracts the first site, applies `to_mcf_ipeps`, and checks the
left-right and up-down density matrix balance.
"""
function local_min_norm_iter(A, params)
    T = A[:,:,:,:,:,1]
    AG = to_mcf_ipeps(T; max_iter=1000, tol=1e-10)

    @tensor Ml[1,6] := AG[1,2,3,4,5] * conj(AG[6,2,3,4,5])
    @tensor Mr[6,3] := AG[1,2,3,4,5] * conj(AG[1,2,6,4,5])

    @tensor Mu[4,6] := AG[1,2,3,4,5] * conj(AG[1,2,3,6,5])
    @tensor Md[6,2] := AG[1,2,3,4,5] * conj(AG[1,6,3,4,5])

    @show norm(Ml - Mr) + norm(Mu - Md)
    return reshape(AG, size(A))
end
