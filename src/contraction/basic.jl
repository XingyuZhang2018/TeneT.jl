# ρmap(ρ, Au::leg3, Ad::leg3) = ein"(dc,csb),dsa -> ab"(ρ,Au,Ad)
# ρmap(ρ, Au::leg4, Ad::leg4) = ein"(dc,cstb),dsta -> ab"(ρ,Au,Ad)
ρmap(ρ, Au::leg3, Ad::leg3) = @tensor out[a,b] := ρ[d,c] * Au[c,s,b] * Ad[d,s,a]
ρmap(ρ, Au::leg4, Ad::leg4) = @tensor out[a,b] := ρ[d,c] * Au[c,s,t,b] * Ad[d,s,t,a]
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
# FLmap(FL, ALu, ALd, M::leg4) = ein"((adf,fgh),dgeb),abc -> ceh"(FL, ALd, M, ALu)
# FLmap(FL, ALu, ALd, M1::leg5, M2::leg5) = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, ALd, M1, M2, ALu)
# FLmap(FL, ALu, ALd, M::leg8) = ein"((aefi,ijkl),efjkghbc),abcd -> dghl"(FL, ALd, M, ALu)
FLmap(FL, ALu, ALd, M::leg4) = @tensor out[c,e,h] := FL[a,d,f] * ALd[f,g,h] * M[d,g,e,b] * ALu[a,b,c]
FLmap(FL, ALu, ALd, M1::leg5, M2::leg5) = @tensor out[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ALu[a,b,c,d]
FLmap(FL, ALu, ALd, M::leg8) = @tensor out[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M[e,f,j,k,g,h,b,c] * ALu[a,b,c,d]

FLmap(FL, ALu, ALd, M::leg5) = FLmap(FL, ALu, ALd, M, conj(M))
FLmap(FL, ALu, ALd, M::Tuple{leg5,leg5}) = FLmap(FL, ALu, ALd, M[1], M[2])

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
# FRmap(FR, ARu, ARd, M::leg4) = ein"((fgh,ceh),dgeb),abc-> adf"(ARd, FR, M, ARu)
# FRmap(FR, ARu, ARd, M1::leg5, M2::leg5) = ein"(((ijkl,dghl),ejgbp),fkhcp),abcd -> aefi"(ARd, FR, M1, M2, ARu)
# FRmap(FR, ARu, ARd, M::leg8) = ein"((ijkl,dghl),efjkghbc), abcd-> aefi"(ARd, FR, M, ARu)
FRmap(FR, ARu, ARd, M::leg4) = @tensor out[a,d,f] := ARd[f,g,h] * FR[c,e,h] * M[d,g,e,b] * ARu[a,b,c]
FRmap(FR, ARu, ARd, M1::leg5, M2::leg5) = @tensor out[a,e,f,i] := ARd[i,j,k,l] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ARu[a,b,c,d]
FRmap(FR, ARu, ARd, M::leg8) = @tensor out[a,e,f,i] := ARd[i,j,k,l] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * ARu[a,b,c,d]

FRmap(FR, ARu, ARd, M::leg5) = FRmap(FR, ARu, ARd, M, conj(M))
FRmap(FR, ARu, ARd, M::Tuple{leg5,leg5}) = FRmap(FR, ARu, ARd, M[1], M[2])

"""
    ```
    ┌── ALuᵢⱼ  ──      ┌──        a──────┬──────c
    Lᵢⱼ   |        =   Lᵢⱼ₊₁      │      │      │
    └── ALdᵢᵣⱼ ──      └──        │      b      │
                                  │      │      │ 
                                  d──────┴──────e               
    ```
"""
# Lmap(Lij, ALuij::leg3, ALdirj::leg3) = ein"(ad,dbe),abc -> ce"(Lij, ALdirj, ALuij)
# Lmap(Lij, ALuij::leg4, ALdirj::leg4) = ein"(ad,dbfe),abfc -> ce"(Lij, ALdirj, ALuij)
Lmap(Lij, ALuij::leg3, ALdirj::leg3) = @tensor out[c,e] := Lij[a,d] * ALdirj[d,b,e] * ALuij[a,b,c]
Lmap(Lij, ALuij::leg4, ALdirj::leg4) = @tensor out[c,e] := Lij[a,d] * ALdirj[d,b,f,e] * ALuij[a,b,f,c]

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
# Rmap(Ri, ARui::leg3, ARdir::leg3) = ein"(abc,ce),dbe->ad"(ARui, Ri, ARdir)
# Rmap(Ri, ARui::leg4, ARdir::leg4) = ein"(abfc,ce),dbfe->ad"(ARui, Ri, ARdir)
Rmap(Ri, ARui::leg3, ARdir::leg3) = @tensor out[a,d] := ARui[a,b,c] * Ri[c,e] * ARdir[d,b,e]
Rmap(Ri, ARui::leg4, ARdir::leg4) = @tensor out[a,d] := ARui[a,b,f,c] * Ri[c,e] * ARdir[d,b,f,e]

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
# ACmap(AC, FL, FR, M::leg4) = ein"((abc,ceh),dgeb),adf -> fgh"(AC,FR,M,FL)
# ACmap(AC, FL, FR, M1::leg5, M2::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),aefi -> ijkl"(AC,FR,M1,M2,FL)
# ACmap(AC, FL, FR, M::leg8) = ein"((abcd,dghl),efjkghbc),aefi -> ijkl"(AC,FR,M,FL)
ACmap(AC, FL, FR, M::leg4) = @tensor out[f,g,h] := AC[a,b,c] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
ACmap(AC, FL, FR, M1::leg5, M2::leg5) = @tensor out[i,j,k,l] := AC[a,b,c,d] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * FL[a,e,f,i]
ACmap(AC, FL, FR, M::leg8) = @tensor out[i,j,k,l] := AC[a,b,c,d] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * FL[a,e,f,i]

ACmap(AC, FL, FR, M::leg5) = ACmap(AC, FL, FR, M, conj(M))
ACmap(AC, FL, FR, M::Tuple{leg5,leg5}) = ACmap(AC, FL, FR, M[1], M[2])



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
# Cmap(C, FL::leg3, FR) = ein"acd,(ab,bce) -> de"(FL,C,FR)
# Cmap(C, FL::leg4, FR) = ein"acde,(ab,bcdf) -> ef"(FL,C,FR)
Cmap(C, FL::leg3, FR) = @tensor out[d,e] := FL[a,c,d] * C[a,b] * FR[b,c,e]
Cmap(C, FL::leg4, FR) = @tensor out[e,f] := FL[a,c,d,e] * C[a,b] * FR[b,c,d,f]

"""
    a ────┬──── c 
    │     b     │ 
    ├─ d ─┼─ e ─┤ 
    │     g     │ 
    f ────┴──── h 
"""
# ACdmap(ACd, FL, FR, M::leg4) = ein"((fgh,ceh),dgeb),adf -> abc"(ACd,FR,M,FL)
# ACdmap(ACd, FL, FR, M1::leg5, M2::leg5) = ein"(((ijkl,dghl),ejgbp),fkhcp),aefi -> abcd"(ACd,FR,M1,M2,FL)
# ACdmap(ACd, FL, FR, M::leg8) = ein"((ijkl,dghl),efjkghbc),aefi -> abcd"(ACd,FR,M,FL)
ACdmap(ACd, FL, FR, M::leg4) = @tensor out[a,b,c] := ACd[f,g,h] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
ACdmap(ACd, FL, FR, M1::leg5, M2::leg5) = @tensor out[a,b,c,d] := ACd[i,j,k,l] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * FL[a,e,f,i]
ACdmap(ACd, FL, FR, M::leg8) = @tensor out[a,b,c,d] := ACd[i,j,k,l] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * FL[a,e,f,i]

ACdmap(ACd, FL, FR, M::leg5) = ACdmap(ACd, FL, FR, M, conj(M))
ACdmap(ACd, FL, FR, M::Tuple{leg5,leg5}) = ACdmap(ACd, FL, FR, M[1], M[2])

# Mmap(AC, ACd, FL, FR) = ein"(abc,ceh),(adf,fgh) -> dgeb"(AC,FR,FL,ACd)
# Mumap(AC, ACd, FL, FR, Mu) = ein"(abcd,dghl),((aefi,ijkl),ejgbp)-> fkhcp"(AC,FR,FL,ACd,Mu)
# Mdmap(AC, ACd, FL, FR, Md) = ein"(abcd,dghl),((aefi,ijkl),fkhcp)-> ejgbp"(AC,FR,FL,ACd,Md)
Mmap(AC, ACd, FL, FR) = @tensor out[d,g,e,b] := AC[a,b,c] * FR[c,e,h] * FL[a,d,f] * ACd[f,g,h]
Mumap(AC, ACd, FL, FR, Mu) = @tensor out[f,k,h,c,p] := AC[a,b,c,d] * FR[d,g,h,l] * FL[a,e,f,i] * ACd[i,j,k,l] * Mu[e,j,g,b,p]
Mdmap(AC, ACd, FL, FR, Md) = @tensor out[e,j,g,b,p] := AC[a,b,c,d] * FR[d,g,h,l] * FL[a,e,f,i] * ACd[i,j,k,l] * Md[f,k,h,c,p]

ALCtoACmap(AL::leg3, C) = @tensor out[a,s,b] := AL[a,s,c] * C[c,b]
ALCtoACmap(AL::leg4, C) = @tensor out[a,s,t,b] := AL[a,s,t,c] * C[c,b]