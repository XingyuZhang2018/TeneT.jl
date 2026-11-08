export exact_free_energy, free_energy

"""
    exact_free_energy(beta; npts=10000)
    exact_free_energy(model::Ising{Square}; npts=10000)

Compute Onsager's exact free energy per site for the square-lattice 2D
classical Ising model using midpoint quadrature.
"""
function exact_free_energy(beta::Real; npts::Integer=10000)
    beta > 0 || throw(ArgumentError("beta must be positive; got $beta"))
    npts > 0 || throw(ArgumentError("npts must be positive; got $npts"))

    s = 0.0
    dt = pi / npts
    @inbounds for i in 1:npts
        t1 = (i - 0.5) * dt
        for j in 1:npts
            t2 = (j - 0.5) * dt
            s += log(cosh(2 * beta)^2 - sinh(2 * beta) * (cos(t1) + cos(t2)))
        end
    end
    s *= dt^2 / (2 * pi^2)
    return -(log(2) + s) / beta
end

exact_free_energy(model::Ising{Square}; kwargs...) = exact_free_energy(model.beta; kwargs...)

function exact_free_energy(model::Ising; kwargs...)
    throw(ArgumentError("exact_free_energy currently supports Ising{Square}; got $(typeof(model.lattice))"))
end

"""
    free_energy(model::Ising, Z; log_power=1)

Convert a partition-function density into free energy per site. Use
`log_power=2` when the transfer eigenvalue represents `Z_per_site^2`, as in
the C4v doubled transfer map.
"""
function free_energy(model::Ising, Z::Number; log_power::Real=1)
    model.beta > 0 || throw(ArgumentError("model.beta must be positive; got $(model.beta)"))
    log_power > 0 || throw(ArgumentError("log_power must be positive; got $log_power"))
    return -real(log(Z)) / (model.beta * log_power)
end

function _free_energy_result(model::Ising, lambda_AC, lambda_C; log_power::Real=1)
    lambda_AC = real(lambda_AC)
    lambda_C = real(lambda_C)
    eigenvalue_ratio = lambda_AC / lambda_C
    Z_per_site = exp(log(eigenvalue_ratio) / log_power)
    return (
        f = free_energy(model, eigenvalue_ratio; log_power),
        lambda_AC = lambda_AC,
        lambda_C = lambda_C,
        eigenvalue_ratio = eigenvalue_ratio,
        Z_per_site = Z_per_site,
    )
end

_c4v_tensor(M::StructArray) = _c4v_local_tensor(M)

"""
    free_energy(rt::C4vVUMPSEnv, M, alg::VUMPS{C4v}, model::Ising)

Estimate the Ising free energy from a converged C4v VUMPS boundary state.
Returns `(f, lambda_AC, lambda_C, Z_per_site)`.
"""
function free_energy(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{C4v}, model::Ising)
    @unpack AL, C, FL = rt
    M_c4v = _c4v_tensor(M)
    AC = ALCtoAC_map(AL, C)
    lambda_AC, _ = ACenv_c4v(AC, FL, M_c4v; alg)
    lambda_C, _ = Cenv_c4v(C, FL; alg)
    return _free_energy_result(model, lambda_AC, lambda_C; log_power=2)
end

"""
    free_energy(rt::VUMPSRuntime, M, alg::VUMPS{General}, model::Ising)

Estimate the Ising free energy from a converged General VUMPS boundary state.
Returns `(f, lambda_AC, lambda_C, Z_per_site)`.
"""
function free_energy(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{General}, model::Ising)
    AC = ALCtoAC(rt.AL, rt.C)
    lambda_ACs, _ = ACenv(AC, rt.FL, M, rt.FR; alg)
    lambda_Cs, _ = Cenv(rt.C, rt.FL, rt.FR; alg)
    return _free_energy_result(model, lambda_ACs[1], lambda_Cs[1])
end
