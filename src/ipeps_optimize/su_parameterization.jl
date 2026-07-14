# Simple Update (SU) parameterization for iPEPS tensors.
# Applies imaginary-time evolution gates via SVD truncation on each bond.

function _su_twosite_hamiltonians(params, d::Int)
    if params.model.lattice isa KagomeOnehole
        throw(ArgumentError("SU_parameterization not yet implemented for $(typeof(params.model.lattice)) (gates would act on the empty site, producing wrong results)"))
    end
    if params.model.lattice isa Honeycomb{:merge}
        throw(ArgumentError("SU_parameterization not yet implemented for Honeycomb{:merge}. Use ifSU=false for merged honeycomb tensors."))
    end
    if params.model.lattice isa Kagome{:merge}
        # d above is the merged physical dim (d_single^3). The Kagome helpers
        # work with the per-sublattice dim, so derive it from the model.
        d_single = Int(2 * params.model.S + 1)
        terms = _heisenberg_bond_terms(params.model, Array; ifrotate=false)
        # Inter-cell H = bond(3→1)+bond(3→2), V = bond(3→1)+bond(2→1)
        function _build_kagome_twosite(sublattice_left, sublattice_right)
            h = zeros(Float64, d, d, d, d)
            for (c, OL, OR) in terms
                OL_d3 = _kagome_site_op(OL, sublattice_left, d_single)
                OR_d3 = _kagome_site_op(OR, sublattice_right, d_single)
                @tensor o[a,b,c,e] := OL_d3[a,b] * OR_d3[c,e]
                h += c * real(o)
            end
            return h
        end
        h_H = _build_kagome_twosite(3, 1) + _build_kagome_twosite(3, 2)
        h_V = _build_kagome_twosite(3, 1) + _build_kagome_twosite(2, 1)

        # Intra-cell (1-2, 2-3) bonds sit on a single merged site. Distribute
        # them symmetrically into the two-site gates as h_onsite⊗I + I⊗h_onsite.
        # Each cell participates in 4 gates per sweep (left/right of H,
        # upper/lower of V), so divide by 4 to recover unit Trotter weight.
        h_onsite = _kagome_onsite_op(terms, 1, 2, d_single, Array) +
                   _kagome_onsite_op(terms, 2, 3, d_single, Array)
        Id = Matrix{Float64}(I, d, d)
        @tensor h_twosite[a,b,c,e] := h_onsite[a,b] * Id[c,e] +
                                      Id[a,b] * h_onsite[c,e]
        h_H += h_twosite / 4
        h_V += h_twosite / 4
    else
        terms = _heisenberg_bond_terms(params.model, Array)
        h = zeros(Float64, d, d, d, d)
        for (c, OL, OR) in terms
            @tensor o[i,j,k,l] := OL[i,j] * OR[k,l]
            h += c * real(o)
        end
        h_H = h_V = h
    end
    return h_H, h_V
end

function _su_twosite_gate(h, τ, d::Int, atype)
    return atype(reshape(exp(-τ * reshape(permutedims(h, (1,3,2,4)), d^2, d^2)), d, d, d, d))
end

function _su_twosite_gates(Aref::AbstractArray{<:Number,5}, params)
    d = size(Aref, 5)
    h_H, h_V = _su_twosite_hamiltonians(params, d)
    atype = _arraytype(Aref)
    return _su_twosite_gate(h_H, params.SUτ, d, atype),
           _su_twosite_gate(h_V, params.SUτ, d, atype)
end

function _su_embed_single_site(A::AbstractArray{<:Number,5}, D_new::Int)
    old_dims = size(A)[1:4]
    d = size(A, 5)
    D_new >= maximum(old_dims) ||
        throw(ArgumentError("single-site SU_parameterization currently supports D_new >= current virtual dimensions; got D_new=$D_new and virtual dims=$old_dims"))

    T = eltype(A)
    A_new = _arraytype(A)(zeros(T, D_new, D_new, D_new, D_new, d))
    A_new[1:old_dims[1], 1:old_dims[2], 1:old_dims[3], 1:old_dims[4], :] = A
    return A_new
end

function _su_expand_structarray(A, D_new::Int)
    return StructArray([_su_embed_single_site(A[p], D_new) for p in 1:length(A)], A.pattern)
end

function _su_orth_basis(M)
    F = svd(M)
    return F.U[:, 1:size(M, 2)]
end

function _su_residual_matrix(M, L0, R0, ::Val{:both_complement})
    QL = _su_orth_basis(L0)
    QR = _su_orth_basis(R0)
    return M - QL * (QL' * M) - (M * QR) * QR' + QL * (QL' * M * QR) * QR'
end

function _su_horizontal_pair_matrix(A_left, A_right, gate)
    @tensor T[f,a,b,j,c,d,e,k] := A_left[a,b,g,f,h] * A_right[g,c,d,e,i] * gate[h,i,j,k]
    return reshape(T, prod(size(T)[1:4]), prod(size(T)[5:8])), size(T)
end

function _su_vertical_pair_matrix(A_upper, A_lower, gate)
    @tensor T[f,a,b,j,c,d,e,k] := A_upper[b,g,f,a,h] * A_lower[c,d,e,g,i] * gate[h,i,j,k]
    return reshape(T, prod(size(T)[1:4]), prod(size(T)[5:8])), size(T)
end

function _su_horizontal_old_subspaces(A_left, A_right, D_old::Int, rows::Int, cols::Int)
    T = promote_type(eltype(A_left), eltype(A_right))
    atype = _arraytype(A_left)
    L0 = atype(zeros(T, rows, D_old))
    R0 = atype(zeros(T, cols, D_old))
    for g in 1:D_old
        L0[:, g] .= vec(permutedims(A_left[:, :, g, :, :], (3, 1, 2, 4)))
        R0[:, g] .= vec(A_right[g, :, :, :, :])
    end
    return L0, R0
end

function _su_vertical_old_subspaces(A_upper, A_lower, D_old::Int, rows::Int, cols::Int)
    T = promote_type(eltype(A_upper), eltype(A_lower))
    atype = _arraytype(A_upper)
    L0 = atype(zeros(T, rows, D_old))
    R0 = atype(zeros(T, cols, D_old))
    for g in 1:D_old
        L0[:, g] .= vec(permutedims(A_upper[:, g, :, :, :], (2, 3, 1, 4)))
        R0[:, g] .= vec(A_lower[:, :, :, g, :])
    end
    return L0, R0
end

function _su_residual_channels(M, L0, R0, nadd::Int; method::Symbol=:both_complement)
    method === :both_complement ||
        throw(ArgumentError("unsupported residual SU growth method $method"))
    R = _su_residual_matrix(M, L0, R0, Val(:both_complement))
    U, S, V = svd(R)
    nadd <= length(S) ||
        throw(ArgumentError("cannot add $nadd residual channels from rank $(length(S)) matrix"))
    sqrtS = sqrt.(S[1:nadd])
    left = U[:, 1:nadd] * Diagonal(sqrtS)
    right = Diagonal(sqrtS) * V'[1:nadd, :]
    return left, right
end

function _su_write_horizontal_growth!(A_left, A_right, left, right, pair_size,
                                      D_old::Int, D_new::Int)
    old = 1:D_old
    new = D_old + 1:D_new
    nadd = D_new - D_old
    L = reshape(left, pair_size[1:4]..., nadd)
    R = reshape(right, nadd, pair_size[5:8]...)
    A_left[old, old, new, old, :] .= permutedims(L[old, old, old, :, :], (2, 3, 5, 1, 4))
    A_right[new, old, old, old, :] .= R[:, old, old, old, :]
    return nothing
end

function _su_write_vertical_growth!(A_upper, A_lower, left, right, pair_size,
                                    D_old::Int, D_new::Int)
    old = 1:D_old
    new = D_old + 1:D_new
    nadd = D_new - D_old
    L = reshape(left, pair_size[1:4]..., nadd)
    R = reshape(right, nadd, pair_size[5:8]...)
    A_upper[old, new, old, old, :] .= permutedims(L[old, old, old, :, :], (3, 5, 1, 2, 4))
    A_lower[old, old, old, new, :] .= permutedims(R[:, old, old, old, :], (2, 3, 4, 1, 5))
    return nothing
end

function _su_residual_horizontal_growth!(Aout_left, Aout_right, Ain_left, Ain_right,
                                         gate, D_old::Int, D_new::Int;
                                         method::Symbol=:both_complement)
    M, pair_size = _su_horizontal_pair_matrix(Ain_left, Ain_right, gate)
    L0, R0 = _su_horizontal_old_subspaces(Ain_left, Ain_right, D_old, size(M)...)
    left, right = _su_residual_channels(M, L0, R0, D_new - D_old; method)
    return _su_write_horizontal_growth!(Aout_left, Aout_right, left, right,
                                        pair_size, D_old, D_new)
end

function _su_residual_vertical_growth!(Aout_upper, Aout_lower, Ain_upper, Ain_lower,
                                       gate, D_old::Int, D_new::Int;
                                       method::Symbol=:both_complement)
    M, pair_size = _su_vertical_pair_matrix(Ain_upper, Ain_lower, gate)
    L0, R0 = _su_vertical_old_subspaces(Ain_upper, Ain_lower, D_old, size(M)...)
    left, right = _su_residual_channels(M, L0, R0, D_new - D_old; method)
    return _su_write_vertical_growth!(Aout_upper, Aout_lower, left, right,
                                      pair_size, D_old, D_new)
end

function _su_residual_growth(A, params; D_new, method::Symbol=:both_complement)
    Ni, Nj = size(A)
    D_old = size(A[1], 1)
    D_new > D_old ||
        throw(ArgumentError("residual SU growth expects D_new > D_old; got D_new=$D_new and D_old=$D_old"))

    Ain = _su_expand_structarray(A, D_new)
    iszero(params.SUτ) && return Ain

    Aout = StructArray(copy.(Ain.data), Ain.pattern)
    exp_h, exp_v = _su_twosite_gates(Ain[1], params)

    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        jr = mod1(j + 1, Nj)
        _su_residual_horizontal_growth!(Aout[i,j], Aout[i,jr],
                                        Ain[i,j], Ain[i,jr],
                                        exp_h, D_old, D_new; method)
    end

    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        ir = mod1(i + 1, Ni)
        _su_residual_vertical_growth!(Aout[i,j], Aout[ir,j],
                                      Ain[i,j], Ain[ir,j],
                                      exp_v, D_old, D_new; method)
    end

    return Aout
end

function SU_parameterization(A::AbstractArray{<:Number,5}, params; D_new,
                             growth::Symbol=:residual,
                             method::Symbol=:both_complement)
    D_old = size(A, 1)
    if D_new == D_old
        return copy(A)
    elseif D_new < D_old
        throw(ArgumentError("single-site SU_parameterization only supports D growth; got D_new=$D_new and D_old=$D_old"))
    end
    return SU_parameterization(StructArray([A], [1;;]), params; D_new, growth, method)[1]
end

"""
    SU_parameterization(A, params; D_new)

Apply one round of Simple Update imaginary-time evolution to the iPEPS tensor array `A`.
First applies horizontal gates, then vertical gates, truncating bond dimensions to `D_new`.

# Arguments
- `A`: StructArray of 5-leg iPEPS tensors with indices (l, d, r, u, p).
- `params`: optimization parameters containing `model` (for the Hamiltonian),
  `SUτ` (imaginary time step), and `pattern` (unit cell layout).
- `D_new`: target bond dimension after SVD truncation.

# Returns
- Updated StructArray of iPEPS tensors after one SU step.

# Algorithm
1. Compute `exp(-τ H)` from the model Hamiltonian.
2. For each site, apply the horizontal gate between site `(i,j)` and `(i,j+1)`,
   SVD-truncate the shared bond to `D_new`.
3. For each site, apply the vertical gate between site `(i,j)` and `(i+1,j)`,
   SVD-truncate the shared bond to `D_new`.
"""
function SU_parameterization(A, params; D_new,
                             growth::Symbol=:residual,
                             method::Symbol=:both_complement)
    Ni, Nj = size(A)
    D, d = size(A[1])[[1,5]]
    if D_new > D && growth === :residual
        return _su_residual_growth(A, params; D_new, method)
    end

    exp_h, exp_v = _su_twosite_gates(A[1], params)

    # Expand bond dimension if needed
    Ah = Zygote.Buffer(A)
    if D_new > D
        for p in 1:length(A)
            Ah[p] = zeros(eltype(A[p]), D_new, D_new, D_new, D_new, d)
            Ah[p][1:D, 1:D, 1:D, 1:D, :] = A[p]
        end
    else
        for p in 1:length(A)
            Ah[p] = A[p]
        end
    end

    # --- Horizontal bonds ---
    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        jr = mod1(j + 1, Nj)

        # Contract two-site tensor with gate: A[i,j] -- exp_h -- A[i,jr]
        # Index convention: Ah[i,j] has indices (l,d,r,u,p) = (a,b,g,f,h)
        #                   Ah[i,jr] has indices (l,d,r,u,p) = (g,c,d,e,i)
        #                   exp_h has indices (h,i,j,k) mapping physical indices
        @tensor AAh_h[f,a,b,j,c,d,e,k] := Ah[i,j][a,b,g,f,h] * Ah[i,jr][g,c,d,e,i] * exp_h[h,i,j,k]
        size_AAh_h = size(AAh_h)

        # SVD truncation
        U, S, V = svd(reshape(AAh_h, prod(size_AAh_h[1:4]), prod(size_AAh_h[5:8])))
        Ah[i,j][:,:,1:D_new,:,:] = permutedims(
            reshape(U[:,1:D_new] * Diagonal(sqrt.(S[1:D_new])), size_AAh_h[1:4]..., D_new),
            (2, 3, 5, 1, 4)
        )
        Ah[i,jr][1:D_new,:,:,:,:] = reshape(
            Diagonal(sqrt.(S[1:D_new])) * V'[1:D_new,:],
            D_new, size_AAh_h[5:8]...
        )
        if D_new < D
            Ah[i,j][:,:,D_new+1:D,:,:] .= 0
            Ah[i,jr][D_new+1:D,:,:,:,:] .= 0
        end
    end
    Ah = copy(Ah)

    # --- Vertical bonds ---
    Av = Zygote.Buffer(Ah)
    for p in 1:length(A)
        Av[p] = Ah[p]
    end

    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        ir = mod1(i + 1, Ni)

        # Contract two-site tensor with gate: Av[i,j] -- exp_h -- Av[ir,j]
        # Index convention: Av[i,j] has indices (l,d,r,u,p) = (b,g,f,a,h)
        #                   Av[ir,j] has indices (l,d,r,u,p) = (c,d,e,g,i)
        @tensor AAh_v[f,a,b,j,c,d,e,k] := Av[i,j][b,g,f,a,h] * Av[ir,j][c,d,e,g,i] * exp_v[h,i,j,k]
        size_AAh_v = size(AAh_v)

        # SVD truncation
        U, S, V = svd(reshape(AAh_v, prod(size_AAh_v[1:4]), prod(size_AAh_v[5:8])))
        Av[i,j][:,1:D_new,:,:,:] = permutedims(
            reshape(U[:,1:D_new] * Diagonal(sqrt.(S[1:D_new])), size_AAh_v[1:4]..., D_new),
            (3, 5, 1, 2, 4)
        )
        Av[ir,j][:,:,:,1:D_new,:] = permutedims(
            reshape(Diagonal(sqrt.(S[1:D_new])) * V'[1:D_new,:], D_new, size_AAh_v[5:8]...),
            (2, 3, 4, 1, 5)
        )
        if D_new < D
            Av[i,j][:,D_new+1:D,:,:,:] .= 0
            Av[ir,j][:,:,:,D_new+1:D,:] .= 0
        end
    end
    Av = copy(Av)

    return Av
end
