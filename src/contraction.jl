ρmap(ρ, Au::leg3, Ad::leg3) = (@tensor ρ[-1; -2] := ρ[3; 1] * Au[1 2; -2] * Ad[-1; 3 2])
ρmap(ρ, Au::leg4, Ad::leg4) = (@tensor ρ[-1; -2] := ρ[4; 1] * Au[1 2 3; -2] * Ad[-1; 4 2 3])
"""
    ρ = ρmap(ρ::Matrix{<:AbstractTensorMap}, A::Matrix{<:AbstractTensorMap})
````
    ┌─ Aᵢⱼ─    ┌─ 
    ρᵢⱼ │   =  ρⱼ₊₁ 
    └─ Aᵢⱼ─    └─
````
"""
function ρmap(ρ, Ai, J::Int)
    Nj = size(Ai, 1)
    @inbounds for j in 1:Nj
        j′ = mod1(J+j-1, Nj)
        ρ = ρmap(ρ, Ai[j′], adjoint(Ai[j′]))
    end
    return ρ
end

"""
    C = LRtoC(L,R)

```
 ── Cᵢⱼ ──  =  ── Lᵢⱼ ── Rᵢⱼ₊₁ ──
```
"""
function LRtoC(L::StructArray, R::StructArray)
    Rijr = circshift(R, (0,-1))
    C = similar(L)
    @inbounds for p in 1:length(L.data)
        i, j = Tuple(findfirst(==(p), L.pattern))
        C[i,j] = L[i,j] * Rijr[i,j]
    end
    return C
end

"""
    AC = ALCtoAC(AL::StructArray, C::StructArray)

```
    ── ACᵢⱼ ──  =  ── ALᵢⱼ ── Cᵢⱼ ──
        |              |
```
"""
function ALCtoAC(AL::StructArray, C::StructArray)
    AC = Zygote.Buffer(AL)
    @inbounds for p in 1:length(AL.data)
        i, j = Tuple(findfirst(==(p), AL.pattern))
        AC[i,j] = AL[i,j] * C[i,j]
    end
    return copy(AC)
end

"""
    FLm = FLmap(FLi::Vector{<:AbstractTensorMap}, 
                ALui::Vector{<:AbstractTensorMap},
                ALdir::Vector{<:AbstractTensorMap}, 
                Ati::Vector{<:AbstractTensorMap}, 
                Abi::Vector{<:AbstractTensorMap})

```
  ┌──       ┌──  ALuᵢⱼ ── 
  │         │     │        
FLᵢⱼ₊₁ =   FLᵢⱼ ─ Oᵢⱼ  ── 
  │         │     │        
  └──       └──  ALdᵢᵣⱼ ─ 
```
"""
function FLmap(FLij::leg3, ALuij::leg3, ALdirj::leg3, Mij::bulk)
    @tensoropt FLij[-1 -2; -3] := FLij[4 3; 1] * ALuij[1 2; -3] * Mij[3 5; -2 2] * ALdirj[-1; 4 5]
    return FLij
end

function FLmap(FLij::leg4, ALuij::leg4, ALdirj::leg4, Mij::ipeps)
    @tensoropt FLij[-1 -2 -3; -4] := FLij[6 5 4; 1] * ALuij[1 2 3; -4] * Mij[5 8 -2 2; 9] * 
                                     conj(Mij[4 7 -3 3; 9]) * ALdirj[-1; 6 8 7]
    return FLij
end

function FLmap(J::Int, FLij::AbstractTensorMap,
               ALui::Vector{<:AbstractTensorMap},
               ALdir::Vector{<:AbstractTensorMap}, 
               Mi::Vector{<:AbstractTensorMap})
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        FLij = FLmap(FLij, ALui[jr], ALdir[jr], Mi[jr])
    end

    return FLij
end

"""
    ```
    ┌── ALuᵢⱼ  ──      ┌──  
    Lᵢⱼ   |        =   Lᵢⱼ₊₁
    └── ALdᵢᵣⱼ ──      └──  
    ```
"""
function Lmap(Li::Vector{<:AbstractTensorMap}, 
              ALui::Vector{<:AbstractTensorMap}, 
              ALdir::Vector{<:AbstractTensorMap})
    Lm = [@tensoropt L[-6; -4] := ALu[1 2 3; -4] * L[5; 1] * ALd[-6; 5 2 3] for (L, ALu, ALd) in zip(Li, ALui, ALdir)]

    return circshift(Lm, 1)
end

"""
    FRm = FRmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                Ati::Vector{<:AbstractTensorMap}, 
                Abi::Vector{<:AbstractTensorMap})

```
    ── ARuᵢⱼ  ──┐          ──┐     
        │       │            │     
    ── Oᵢⱼ   ──FRᵢⱼ  =    ──FRᵢⱼ₋₁ 
        │       │            │     
    ── ARdᵢᵣⱼ ──┘          ──┘     
```
"""
function FRmap(FRij::leg3, ARuij::leg3, ARdirj::leg3, Mij::bulk)
    @tensoropt FRij[-1 -2; -3] := ARuij[-1 1; 2] * FRij[2 3; 5] * Mij[-2 4; 3 1] * ARdirj[5; -3 4]
    return FRij
end

function FRmap(FRij::leg4, ARuij::leg4, ARdirj::leg4, Mij::ipeps)
    @tensoropt FRij[-1 -2 -3; -4] := ARuij[-1 1 2; 3] * FRij[3 4 5; 8] * Mij[-2 7 4 1; 9] * 
                                     conj(Mij[-3 6 5 2; 9]) * ARdirj[8; -4 7 6]
    return FRij
end

function FRmap(J::Int, FRij::AbstractTensorMap, 
               ARui::Vector{<:AbstractTensorMap}, 
               ARdir::Vector{<:AbstractTensorMap}, 
               Mi::Vector{<:AbstractTensorMap})

    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        FRij = FRmap(FRij, ARui[jr], ARdir[jr], Mi[jr])
    end

    return FRij
end

"""
    Rm = Rmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                )

```
    ── ARuᵢⱼ  ──┐          ──┐    
        │       Rᵢⱼ  =       Rᵢⱼ₋₁  
    ── ARdᵢᵣⱼ ──┘          ──┘     
```
"""
function Rmap(Rij::AbstractTensorMap, 
              ARuij::AbstractTensorMap, 
              ARdirj::AbstractTensorMap)
    @tensoropt Rij[-1; -5] := ARuij[-1 2 3; 4] * Rij[4; 6] * ARdirj[6; -5 2 3] 

    return Rij
end

function Rmap(J, Rij::AbstractTensorMap, 
              ARui::Vector{<:AbstractTensorMap}, 
              ARdir::Vector{<:AbstractTensorMap})
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        Rij = Rmap(Rij, ARui[jr], ARdir[jr])
    end

    return Rij
end


"""
    ACm = ACmap(ACj::Vector{<:AbstractTensorMap}, 
                FLj::Vector{<:AbstractTensorMap}, 
                FRj::Vector{<:AbstractTensorMap},
                Atj::Vector{<:AbstractTensorMap},
                Abj::Vector{<:AbstractTensorMap})

```
                                ┌─────── ACᵢⱼ ─────┐       
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │      
│        │         │      =     FLᵢⱼ ─── Oᵢⱼ ───── FRᵢⱼ   
                                │        │         │      
                                                                
```
"""
function ACmap(ACij::leg3, FLij::leg3, FRij::leg3, Mij::bulk)
    @tensoropt ACij[-1 -2; -3] := ACij[1 2; 3] * FLij[-1 4; 1]* Mij[4 -2; 5 2] * FRij[3 5; -3] 
    return ACij
end

function ACmap(ACij::leg4, FLij::leg4, FRij::leg4, Mij::ipeps)
    @tensoropt ACij[-1 -2 -3; -4] := ACij[1 2 3; 4] * FLij[-1 6 5; 1]* Mij[6 -2 7 2; 9] * 
                                     conj(Mij[5 -3 8 3; 9]) * FRij[4 7 8; -4] 
    return ACij
end

function ACmap(I::Int, ACij::AbstractTensorMap, 
               FLj::Vector{<:AbstractTensorMap}, 
               FRj::Vector{<:AbstractTensorMap},
                Mj::Vector{<:AbstractTensorMap})
    Ni = length(FLj)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        ACij = ACmap(ACij, FLj[ir], FRj[ir], Mj[ir])
    end

    return ACij
end
"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐       
┌── Cᵢ₊₁ⱼ ──┐       │           │       
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ     
                    │           │       
                                                                       
```
"""
function Cmap(Cij::AbstractTensorMap, FLijr::leg3, FRij::leg3)
    @tensoropt Cij[-1; -2] := Cij[1; 2] * FLijr[-1 3; 1] * FRij[2 3; -2]
    return Cij
end

function Cmap(Cij::AbstractTensorMap, FLijr::leg4, FRij::leg4)
    @tensoropt Cij[-1; -2] := Cij[1; 2] * FLijr[-1 3 4; 1] * FRij[2 3 4; -2]
    return Cij
end

function Cmap(I, Cij::AbstractTensorMap,
              FLjr::Vector{<:AbstractTensorMap}, 
              FRj::Vector{<:AbstractTensorMap})
    Ni = length(FLjr)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        Cij = Cmap(Cij, FLjr[ir], FRj[ir])
    end

    return Cij
end

# """
#     nearest_neighbour_energy(ipeps::InfinitePEPS, Hh, Hv, env::VUMPSEnv)

#     Compute the energy of the nearest neighbour Hamiltonian for an infinite PEPS.

# ```
#         ┌────── ACuᵢⱼ ────ARuᵢᵣ ──────┐   
#         │       │          │          │   
#         FLoᵢⱼ ── Oᵢⱼ ────── Oᵢᵣ ──── FRoᵢᵣ     ir = Ni + 1 - i
#         │       │          │          │        jr = j + 1    
#         └───── ACdᵢᵣⱼ ─────ARdᵢᵣᵣ─────┘           

#         ┌─────── ACuᵢⱼ ─────┐    
#         │         │         │    
#         FLuᵢⱼ ─── Oᵢⱼ ───── FRuᵢⱼ               ir = i + 1
#         │         │         │                  irr = Ni - i
#         FLoᵢᵣⱼ ── Oᵢᵣⱼ ───  FRoᵢᵣⱼ    
#         │         │         │    
#         └──────  ACdᵢᵣᵣⱼ ───┘
# ```
# """
# function nearest_neighbour_energy(ipeps::InfinitePEPS, Hh, Hv, env::VUMPSEnv)
#     @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
#     Ni, Nj = size(ipeps)

#     energy_tol = 0
#     for j in 1:Nj, i in 1:Ni
#         # horizontal contraction
#         ir = Ni + 1 - i
#         jr = mod1(j + 1, Nj)
#         @tensoropt oph[-1 -2; -3 -4] := FLo[i,j][18 12 15; 5] * ACu[i,j][5 6 7; 8] * ipeps.A[i,j][-1; 6 13 19 12] * 
#                                         conj(ipeps.A[i,j][-3; 7 16 20 15]) * conj(ACd[ir,j][18 19 20; 21]) * 
#                                         ARu[i,jr][8 9 10; 11] * ipeps.A[i,jr][-2; 9 14 22 13] * 
#                                         conj(ipeps.A[i,jr][-4; 10 17 23 16]) * conj(ARd[ir,jr][21 22 23; 24]) * FRo[i,jr][11 14 17; 24]

#         @tensor eh = oph[1 2; 3 4] * Hh[3 4; 1 2]
#         @tensor nh = oph[1 2; 1 2]
#         energy_tol += eh / nh
#         # @show eh / nh eh nh
        
#         # vertical contraction
#         ir = mod1(i + 1, Ni)
#         irr = mod1(Ni - i, Ni)
#         @tensoropt opv[-1 -2; -3 -4] := FLu[i,j][21 19 20; 18] * ACu[i,j][18 12 15 5] * ipeps.A[i,j][-1; 12 6 13 19] * 
#                                     conj(ipeps.A[i,j][-3; 15 7 16 20]) * FRu[i,j][5 6 7; 8] * FLo[ir,j][24 22 23; 21] * 
#                                     ipeps.A[ir,j][-2; 13 9 14 22] * conj(ipeps.A[ir,j][-4; 16 10 17 23]) * 
#                                     FRo[ir,j][8 9 10; 11] * conj(ACd[irr,j][24 14 17; 11])

#         @tensor ev = opv[1 2; 3 4] * Hv[3 4; 1 2]
#         @tensor nv = opv[1 2; 1 2]
#         energy_tol += ev / nv 
#         # @show ev / nv ev nv

#         # penalty term 
#         # energy_tol += 0.1 * abs(eh / nh - eh / nh)
#     end

#     return energy_tol/Ni/Nj
# end