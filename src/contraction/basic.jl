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
FLmap(FL, ALu, ALd, M1::leg5, M2::leg5) = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, ALd, M1, M2, ALu)
FLmap(FL, ALu, ALd, M::leg8) = ein"((aefi,ijkl),efjkghbc),abcd -> dghl"(FL, ALd, M, ALu)

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
FRmap(FR, ARu, ARd, M::leg4) = ein"((fgh,ceh),dgeb),abc-> adf"(ARd, FR, M, ARu)
FRmap(FR, ARu, ARd, M1::leg5, M2::leg5) = ein"(((ijkl,dghl),ejgbp),fkhcp),abcd -> aefi"(ARd, FR, M1, M2, ARu)
FRmap(FR, ARu, ARd, M::leg8) = ein"((ijkl,dghl),efjkghbc), abcd-> aefi"(ARd, FR, M, ARu)

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
Lmap(Lij, ALuij::leg3, ALdirj::leg3) = ein"(ad,dbe),abc -> ce"(Lij, ALdirj, ALuij)

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
ACmap(AC, FL, FR, M1::leg5, M2::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),aefi -> ijkl"(AC,FR,M1,M2,FL)
ACmap(AC, FL, FR, M::leg8) = ein"((abcd,dghl),efjkghbc),aefi -> ijkl"(AC,FR,M,FL)

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
Cmap(C, FL::leg3, FR) = ein"acd,(ab,bce) -> de"(FL,C,FR)
Cmap(C, FL::leg4, FR) = ein"acde,(ab,bcdf) -> ef"(FL,C,FR)