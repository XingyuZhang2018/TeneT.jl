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
FLmap(FL, ALu, ALd, M::leg8) = ein"((aefi,ijkl),efjkghbc),abcd -> dghl"(FL, ALd, M, ALu)

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
FRmap(FR, ARu, ARd, M::leg8) = ein"((abcd,dghl),efjkghbc),ijkl -> aefi"(ARu, FR, M, ARd)

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
ACmap(AC, FL, FR, M::leg8) = ein"((abcd,dghl),efjkghbc),aefi -> ijkl"(AC,FR,M,FL)

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
