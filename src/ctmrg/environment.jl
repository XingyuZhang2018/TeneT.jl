struct CTMEnv{:honeycomb}
    Ta::AbstractArray
    Tb::AbstractArray
    C::AbstractArray
end

function ctmrg_get_isometries(Ta, Tb, C, M, alg::CTMRG{:honeycomb})
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    processed_indices = Set{Int}()
    Ni, Nj = size(M[1])
    Ma, Mb = M
    χ, D = size(Ta[1][1])[[1,2]]
    Ua = Zygote.Buffer(Ta)
    Ub = Zygote.Buffer(Tb)
    for j in 1:Nj, i in 1:Ni
        p = M[1].pattern[i,j]
        for k in 1:3
            kr = mod1(k-1,3)
            C_enlarge = Cmap_enlarge(C[k], Ta[kr][p], Tb[k][p], Ma[p], Mb[p])
            F = svd(reshape(C_enlarge, χ*D^2, χ*D^2))
            Ua[k][p] = reshape(F.Vt[1:χ, :], χ, D, D, χ)
            Ub[kr][p] = reshape(F.U[:, 1:χ], χ, D, D, χ)
        end
        push!(processed_indices, p)
        if length(processed_indices) == length(Ta[1].data)
            break
        end
    end

    return copy(Ua), copy(Ub)
end

function T_update(Ta, Tb, Ua, Ub, M, alg::CTMRG{:honeycomb})
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    processed_indices = Set{Int}()
    Ni, Nj = size(M[1])
    Ma, Mb = M
    Ta′ = Zygote.Buffer(Ta)
    Tb′ = Zygote.Buffer(Tb)
    for j in 1:Nj, i in 1:Ni
        p = M[1].pattern[i,j]
        for k in 1:3
            perm = (mod1(2-k, 3), mod1(3-k, 3), mod1(4-k, 3))
            Tb′[k][p] = Tmap(Ua[k][p], Ta[k][p], permutedims(Ma[p], perm))
            Ta′[k][p] = Tmap(Tb[k][p], Ub[k][p], permutedims(Mb[p], perm))
        end
        push!(processed_indices, p)
        if length(processed_indices) == length(Ta[1].data)
            break
        end
    end
    return copy(Ta′), copy(Tb′)
end

function  C_update(C, Ta, Tb, alg::CTMRG{:honeycomb})
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    processed_indices = Set{Int}()
    Ni, Nj = size(C[1])
    C′ = Zygote.Buffer(C)
    for j in 1:Nj, i in 1:Ni
        p = C[1].pattern[i,j]
        for k in 1:3
            kr = mod1(k-1,3)
            C′[k][p] = Cmap_ctmrg(C[k][p], Ta[k][p], Tb[kr][p])
        end
        push!(processed_indices, p)
        if length(processed_indices) == length(Ta[1].data)
            break
        end
    end
    return copy(C′)
end