export optimize_clifford

"""
    optimize_clifford(model::Kitaev{Honeycomb{:brickwall}};
                      n_layers::Int=3, max_sweeps::Int=10,
                      tol::Real=1e-8, verbosity::Int=1)

Greedy sweep Clifford optimizer for the Kitaev model on the honeycomb brickwall lattice.

Searches over the 720-element two-qubit Clifford group to find unitaries that
minimize the total bond entanglement entropy of the Hamiltonian, layer by layer.

Returns a Dict with keys:
- `:h_transformed` => tuple of 3 bare bond operators (coupling constants divided out)
- `:circuit` => Vector of `(C_matrix, bond_type_symbol, site_pair_tuple)` recording applied Cliffords
- `:entanglement_history` => Vector of total entanglement after each sweep
"""
function optimize_clifford(model::Kitaev{Honeycomb{:brickwall}};
                           n_layers::Int=3, max_sweeps::Int=10,
                           tol::Real=1e-8, verbosity::Int=1)
    @unpack Jx, Jy, Jz = model

    # 1. Get bare Hamiltonian terms: (hx, hy, hz)
    h = hamiltonian(model)

    # 2. Apply coupling constants to work with the physical Hamiltonian
    h_current = [ComplexF64.(Jx * h[1]),
                 ComplexF64.(Jy * h[2]),
                 ComplexF64.(Jz * h[3])]

    # 3. Generate the full two-qubit Clifford group (720 elements)
    cliffords = generate_clifford_group()

    # Bond type iteration order: z first (intra-cell in brickwall), then x, y
    bond_types = [:z, :x, :y]
    # Map bond type to index in h_current: x->1, y->2, z->3
    bond_idx_map = Dict(:x => 1, :y => 2, :z => 3)
    # Site pair (both sites in minimal 2-site unit cell)
    site_pair = (1, 2)

    circuit = Vector{Tuple{Matrix{ComplexF64}, Symbol, Tuple{Int,Int}}}()
    entanglement_history = Float64[]

    S_prev = sum(bond_entanglement_entropy(h_current[i]) for i in 1:3)

    if verbosity >= 1
        @printf("Initial total entanglement: %.10f\n", S_prev)
    end

    for sweep in 1:max_sweeps
        for layer in 1:n_layers
            for bond_type in bond_types
                bidx = bond_idx_map[bond_type]

                # Current bond operator for this bond type
                h_bond = h_current[bidx]

                # Search over all 720 Cliffords for the one that minimizes
                # the entanglement of this bond
                best_S = bond_entanglement_entropy(h_bond)
                best_C = nothing

                for C in cliffords
                    h_trial = transform_bond_hamiltonian(h_bond, C)
                    S_trial = bond_entanglement_entropy(h_trial)
                    if S_trial < best_S - 1e-14
                        best_S = S_trial
                        best_C = C
                    end
                end

                # If we found a Clifford that improves things, apply it to ALL bonds
                if best_C !== nothing
                    for idx in 1:3
                        h_current[idx] = transform_bond_hamiltonian(h_current[idx], best_C)
                    end
                    push!(circuit, (best_C, bond_type, site_pair))

                    if verbosity >= 2
                        S_now = sum(bond_entanglement_entropy(h_current[i]) for i in 1:3)
                        @printf("  Sweep %d, layer %d, bond %s: total S = %.10f\n",
                                sweep, layer, string(bond_type), S_now)
                    end
                end
            end
        end

        # Record total entanglement after this sweep
        S_total = sum(bond_entanglement_entropy(h_current[i]) for i in 1:3)
        push!(entanglement_history, S_total)

        if verbosity >= 1
            @printf("Sweep %d: total entanglement = %.10f\n", sweep, S_total)
        end

        # Check convergence
        improvement = S_prev - S_total
        S_prev = S_total
        if improvement < tol
            if verbosity >= 1
                @printf("Converged after %d sweeps (improvement %.2e < tol %.2e)\n",
                        sweep, improvement, tol)
            end
            break
        end
    end

    # 5. Divide back by coupling constants to get bare operators
    couplings = [Jx, Jy, Jz]
    h_transformed = Tuple(
        abs(couplings[i]) > 1e-15 ? real.(_mat_to_tensor(_tensor_to_mat(h_current[i]) / couplings[i])) : real.(h_current[i])
        for i in 1:3
    )

    return Dict(
        :h_transformed => h_transformed,
        :circuit => circuit,
        :entanglement_history => entanglement_history,
    )
end
