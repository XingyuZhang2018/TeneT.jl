# Simple Update (SU) parameterization for iPEPS tensors.
# Applies imaginary-time evolution gates via SVD truncation on each bond.

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
function SU_parameterization(A, params; D_new)
    Ni, Nj = size(A)
    D, d = size(A[1])[[1,5]]
    if params.model.lattice isa KagomeOnehole
        throw(ArgumentError("SU_parameterization not yet implemented for $(typeof(params.model.lattice)) (gates would act on the empty site, producing wrong results)"))
    end
    if params.model.lattice isa Kagome{:merge}
        terms = _heisenberg_bond_terms(params.model, Array; ifrotate=false)
        # Inter-cell H = bond(3→1)+bond(3→2), V = bond(3→1)+bond(2→1)
        function _build_kagome_twosite(sublattice_left, sublattice_right)
            h = zeros(Float64, d^3, d^3, d^3, d^3)
            for (c, OL, OR) in terms
                OL_d3 = _kagome_site_op(OL, sublattice_left, d)
                OR_d3 = _kagome_site_op(OR, sublattice_right, d)
                @tensor o[a,b,c,d] := OL_d3[a,b] * OR_d3[c,d]
                h += c * real(o)
            end
            return h
        end
        h_H = _build_kagome_twosite(3, 1) + _build_kagome_twosite(3, 2)
        h_V = _build_kagome_twosite(3, 1) + _build_kagome_twosite(2, 1)
        h_onsite = _kagome_onsite_op(terms, 1, 2, d, Array) + _kagome_onsite_op(terms, 2, 3, d, Array)
        @tensor h_twosite[1,2,3,4] := h_onsite[1,2] * h_onsite[3,4]
        h_H += h_twosite
        h_V += h_twosite
    else
        terms = _heisenberg_bond_terms(params.model, Array)
        h = zeros(Float64, d, d, d, d)
        for (c, OL, OR) in terms
            @tensor o[i,j,k,l] := OL[i,j] * OR[k,l]
            h += c * real(o)
        end
        h_H = h_V = h
    end

    exp_h = _arraytype(A[1])(reshape(exp(-params.SUτ * reshape(permutedims(h_H, (1,3,2,4)), d^2, d^2)), d, d, d, d))

    # Expand bond dimension if needed
    Ah = Zygote.Buffer(A)
    if D_new > D
        for p in 1:length(A)
            Ah[p] = zeros(ComplexF64, D_new, D_new, D_new, D_new, d)
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

    exp_h = _arraytype(A[1])(reshape(exp(-params.SUτ * reshape(permutedims(h_V, (1,3,2,4)), d^2, d^2)), d, d, d, d))

    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        ir = mod1(i + 1, Ni)

        # Contract two-site tensor with gate: Av[i,j] -- exp_h -- Av[ir,j]
        # Index convention: Av[i,j] has indices (l,d,r,u,p) = (b,g,f,a,h)
        #                   Av[ir,j] has indices (l,d,r,u,p) = (c,d,e,g,i)
        @tensor AAh_v[f,a,b,j,c,d,e,k] := Av[i,j][b,g,f,a,h] * Av[ir,j][c,d,e,g,i] * exp_h[h,i,j,k]
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
