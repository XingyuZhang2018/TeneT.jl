ρmap(ρ, Au::leg3, Ad::leg3) = ein"(dc,csb),dsa -> ab"(ρ,Au,Ad)
ρmap(ρ, Au::leg4, Ad::leg4) = ein"(dc,cstb),dsta -> ab"(ρ,Au,Ad)

function ρmap(ρ, Ai, J::Int)
    Nj = size(Ai,1)
    for j = 1:Nj
        jr = mod1(J+j-1, Nj)
        ρ = ρmap(ρ,Ai[jr],conj(Ai[jr]))
    end
    return ρ
end

"""
    FLm = FLmap(ALu, ALd, M, FL)

```
  ┌──       ┌──  ALuᵢⱼ  ──                     a ────┬──── c 
  │         │     │                            │     b     │ 
FLᵢⱼ₊₁ =   FLᵢⱼ ─ Mᵢⱼ   ──                     ├─ d ─┼─ e ─┤ 
  │         │     │                            │     g     │ 
  └──       └──  ALdᵢᵣⱼ  ─                     f ────┴──── h 
```
"""
FLmap(FL, ALu, ALd, M::leg4) = ein"((adf,fgh),dgeb),abc -> ceh"(FL, ALd, M, ALu)
FLmap(FL, ALu, ALd, M::leg5) = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, ALd, M, conj(M), ALu)
FLmap(FL, ALu, ALd, M1::leg5, M2::leg5) = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, ALd, M1, M2, ALu)
FLmap(FL, ALu, ALd, M::leg8) = ein"((aefi,ijkl),efjkghbc),abcd -> dghl"(FL, ALd, M, ALu)

function FLmap(J::Int, FLij, ALui, ALdir, Mi; ifcheckpoint=false, forloop_iter=1)
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        FLij = ifcheckpoint ? checkpoint(FLmap_forloop, FLij, ALui[jr], ALdir[jr], Mi[jr]; forloop_iter) : FLmap_forloop(FLij, ALui[jr], ALdir[jr], Mi[jr]; forloop_iter)
    end
    return FLij
end

"""
    FRm = FRmap(ARu, ARd, M, FR, i)

```
    ── ARuᵢⱼ  ──┐          ──┐          a ────┬──── c 
        │       │            │          │     b     │ 
    ── Mᵢⱼ   ──FRᵢⱼ  =    ──FRᵢⱼ₋₁      ├─ d ─┼─ e ─┤ 
        │       │            │          │     g     │ 
    ── ARdᵢᵣⱼ ──┘          ──┘          f ────┴──── h 
```
"""
FRmap(FR, ARu, ARd, M::leg4) = ein"((abc,ceh),dgeb),fgh -> adf"(ARu, FR, M, ARd)
FRmap(FR, ARu, ARd, M::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),ijkl -> aefi"(ARu, FR, M, conj(M), ARd)
FRmap(FR, ARu, ARd, M1::leg5, M2::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),ijkl -> aefi"(ARu, FR, M1, M2, ARd)
FRmap(FR, ARu, ARd, M::leg8) = ein"((abcd,dghl),efjkghbc),ijkl -> aefi"(ARu, FR, M, ARd)

function FRmap(J::Int, FRij, ARui, ARdir, Mi; ifcheckpoint=false, forloop_iter=1)
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        FRij = ifcheckpoint ? checkpoint(FRmap_forloop, FRij, ARui[jr], ARdir[jr], Mi[jr]; forloop_iter) : FRmap_forloop(FRij, ARui[jr], ARdir[jr], Mi[jr]; forloop_iter)
    end
    return FRij
end

"""
    ```
    ┌── ALuᵢⱼ  ──      ┌──        a──────┬──────c
    Lᵢⱼ   |        =   Lᵢⱼ₊₁      │      │      │
    └── ALdᵢᵣⱼ ──      └──        │      b      │
                                  │      │      │ 
                                  d──────┴──────e               
    ```
"""
Lmap(Lij, ALuij::leg3, ALdirj::leg3) = ein"(ad,dbe),abc -> ce"(Lij, ALdirj, ALuij)

function Lmap(J::Int, Lij, ALui, ALdir)
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        Lij = Lmap(Lij, ALui[jr], ALdir[jr])
    end
    return Lij
end

"""
    Rm = Rmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                )

```
    ── ARuᵢⱼ  ──┐          ──┐           a──────┬──────c    
        │       Rᵢⱼ  =       Rᵢⱼ₋₁       │      │      │ 
    ── ARdᵢᵣⱼ ──┘          ──┘           │      b      │    
                                         │      │      │      
                                         d──────┴──────e   
```
"""
Rmap(Ri, ARui::leg3, ARdir::leg3) = ein"(abc,ce),dbe->ad"(ARui, Ri, ARdir)
function Rmap(J::Int, Rij, ARui, ARdir)
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        Rij = Rmap(Rij, ARui[jr], ARdir[jr])
    end
    return Rij
end


"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐              a ────┬──── c  
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │              │     b     │ 
│        │         │      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ           ├─ d ─┼─ e ─┤ 
                                │        │         │              │     g     │ 
                                                                  f ────┴──── h 
                                                               
```
"""
ACmap(AC, FL, FR, M::leg4) = ein"((abc,ceh),dgeb),adf -> fgh"(AC,FR,M,FL)
ACmap(AC, FL, FR, M::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),aefi -> ijkl"(AC,FR,M,conj(M),FL)
ACmap(AC, FL, FR, M1::leg5, M2::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),aefi -> ijkl"(AC,FR,M1,M2,FL)
ACmap(AC, FL, FR, M::leg8) = ein"((abcd,dghl),efjkghbc),aefi -> ijkl"(AC,FR,M,FL)

function ACmap(I::Int, ACij, FLj, FRj, Mj; ifcheckpoint=false, forloop_iter=1)
    Ni = length(Mj)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        ACij = ifcheckpoint ? checkpoint(ACmap_forloop, ACij, FLj[ir], FRj[ir], Mj[ir]; forloop_iter) : ACmap_forloop(ACij, FLj[ir], FRj[ir], Mj[ir]; forloop_iter)
    end
    return ACij
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐            a ─── b
┌── Cᵢ₊₁ⱼ ──┐       │           │            │     │
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ          ├─ c ─┤
                    │           │            │     │
                                             d ─── e                                    
```
"""
Cmap(C, FL::leg3, FR) = ein"acd,(ab,bce) -> de"(FL,C,FR)
Cmap(C, FL::leg4, FR) = ein"acde,(ab,bcdf) -> ef"(FL,C,FR)

function Cmap(I, Cij, FLjr, FRj)
    Ni = length(FLjr)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        Cij = Cmap(Cij, FLjr[ir], FRj[ir])
    end
    return Cij
end