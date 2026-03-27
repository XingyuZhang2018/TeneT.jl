# QRCTM boundary algorithm
#
# QR-based Corner Transfer Matrix method.
# Uses QR decomposition of the combined C*T tensor to obtain the projector U,
# then applies a single transfer-matrix step followed by a QR-based corner update.

# ── QRCTM-specific getU ────────────────────────────────────────────────

"""
    getU(env::CTMEnv, ::QRCTM)

Compute the isometric projector U and updated corner R from a QR
decomposition of C*T.
"""
function getU(env::CTMEnv, ::QRCTM)
    C = env.C
    T = env.T
    Tu = CTtoT(C, T)
    # _to_tail flattens all-but-first dimensions into rows, first dim into columns:
    #   (chi, D, chi) -> (D*chi, chi)
    U, Cnew = qr_for_ad(_to_tail(Tu))
    U = reshape(U, size(T))
    return U, Cnew
end

# ── QRCTM leftmove ─────────────────────────────────────────────────────

"""
    leftmove(M, env::CTMEnv, alg::QRCTM)

One CTM left-move step for the QRCTM algorithm.

1. QR-decompose C*T to get projector U and remainder R
2. Apply the transfer matrix to get new edge tensor T
3. Update corner via Cmap with R, new T, and U
4. Return updated environment and convergence error
"""
function leftmove(M, env::CTMEnv, alg::QRCTM)
    C = env.C
    T = env.T

    CT = _to_tail(CTtoT(C, T))
    U, R = qr_for_ad(CT)
    U = reshape(U, size(T))

    T = FLmap_parallel(T, U, U, M;
                       ifparallel=alg.ifparallel,
                       forloop_iter=alg.forloop_iter)
    C_new = Cmap(R, T, U)

    T /= Zygote.@ignore norm(T)
    C_new /= Zygote.@ignore norm(C_new)
    err = Zygote.@ignore norm(C_new - C)

    return CTMEnv(C_new, T), err
end
