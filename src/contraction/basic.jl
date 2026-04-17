"""
    _downcast_eltype(T, A) -> A'

Cast `A`'s element type to the smallest type compatible with real scalar type `T`
while preserving complex-ness. Used to lower precision inside FLmap/FRmap/ACmap
when `VUMPS.inner_etype` is set.

Rules:
- `T === nothing`               → return `A` unchanged (identity)
- `T === real(eltype(A))`       → return `A` unchanged (identity; no copy)
- `eltype(A) <: Complex`        → cast to `Complex{T}`
- otherwise                     → cast to `T`
"""
_downcast_eltype(::Nothing, A) = A
function _downcast_eltype(T::Type, A)
    Ta = eltype(A)
    if Ta <: Complex
        T === real(Ta) && return A
        return Complex{T}.(A)
    else
        T === Ta && return A
        return T.(A)
    end
end

ALCtoAC_map(AL::leg3, C) = @tensor result[a,b,d] := AL[a,b,c] * C[c,d]
ALCtoAC_map(AL::leg4, C) = @tensor result[a,b,c,e] := AL[a,b,c,d] * C[d,e]
CTtoT(C, T::leg3) = @tensor T[a,c,d] := C[a,b] * T[b,c,d]
CTtoT(C, T::leg4) = @tensor T[a,c,d,e] := C[a,b] * T[b,c,d,e]
CTCtoT(C, T::leg3) = @tensor T[a,c,e] := C[a,b] * T[b,c,d] * C[d,e]
CTCtoT(C, T::leg4) = @tensor T[a,c,d,f] := C[a,b] * T[b,c,d,e] * C[e,f]

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
function FLmap(FL, ALu, ALd, M::leg4)
    @tensor result[c,e,h] := FL[a,d,f] * ALd[f,g,h] * M[d,g,e,b] * ALu[a,b,c]
    return result
end
function FLmap(FL, ALu, ALd, M1::leg5, M2::leg5; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ALu[a,b,c,d]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M1_t  = _downcast_eltype(inner_etype, M1)
        M2_t  = _downcast_eltype(inner_etype, M2)
        @tensor result_t[d,g,h,l] := FL_t[a,e,f,i] * ALd_t[i,j,k,l] * M1_t[e,j,g,b,p] * M2_t[f,k,h,c,p] * ALu_t[a,b,c,d]
        return T_out.(result_t)
    end
end
function FLmap(FL, ALu, ALd, M::leg8)
    @tensor result[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M[e,f,j,k,g,h,b,c] * ALu[a,b,c,d]
    return result
end

FLmap(FL, ALu, ALd, M::leg5; inner_etype=nothing) =
    FLmap(FL, ALu, ALd, M, conj(M); inner_etype)
FLmap(FL, ALu, ALd, M::Tuple{leg5,leg5}; inner_etype=nothing) =
    FLmap(FL, ALu, ALd, M[1], M[2]; inner_etype)

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
function FRmap(FR, ARu, ARd, M::leg4)
    @tensor result[a,d,f] := ARd[f,g,h] * FR[c,e,h] * M[d,g,e,b] * ARu[a,b,c]
    return result
end
function FRmap(FR, ARu, ARd, M1::leg5, M2::leg5)
    @tensor result[a,e,f,i] := ARd[i,j,k,l] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ARu[a,b,c,d]
    return result
end
function FRmap(FR, ARu, ARd, M::leg8)
    @tensor result[a,e,f,i] := ARd[i,j,k,l] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * ARu[a,b,c,d]
    return result
end

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
function Lmap(Lij, ALuij::leg3, ALdirj::leg3)
    @tensor result[c,e] := Lij[a,d] * ALdirj[d,b,e] * ALuij[a,b,c]
    return result
end

function Lmap(Lij, ALuij::leg4, ALdirj::leg4)
    @tensor result[c,e] := Lij[a,d] * ALdirj[d,b,f,e] * ALuij[a,b,f,c]
    return result
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
function Rmap(Ri, ARui::leg3, ARdir::leg3)
    @tensor result[a,d] := ARui[a,b,c] * Ri[c,e] * ARdir[d,b,e]
    return result
end

function Rmap(Ri, ARui::leg4, ARdir::leg4)
    @tensor result[a,d] := ARui[a,b,f,c] * Ri[c,e] * ARdir[d,b,f,e]
    return result
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
function ACmap(AC, FL, FR, M::leg4)
    @tensor result[f,g,h] := AC[a,b,c] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
    return result
end
function ACmap(AC, FL, FR, M1::leg5, M2::leg5)
    @tensor result[i,j,k,l] := AC[a,b,c,d] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * FL[a,e,f,i]
    return result
end
function ACmap(AC, FL, FR, M::leg8)
    @tensor result[i,j,k,l] := AC[a,b,c,d] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * FL[a,e,f,i]
    return result
end

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
function Cmap(C, FL::leg3, FR)
    @tensor result[d,e] := FL[a,c,d] * C[a,b] * FR[b,c,e]
    return result
end
function Cmap(C, FL::leg4, FR)
    @tensor result[e,f] := FL[a,c,d,e] * C[a,b] * FR[b,c,d,f]
    return result
end

"""
    a ────┬──── c
    │     b     │
    ├─ d ─┼─ e ─┤
    │     g     │
    f ────┴──── h
"""
function ACdmap(ACd, FL, FR, M::leg4)
    @tensor result[a,b,c] := ACd[f,g,h] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
    return result
end
function ACdmap(ACd, FL, FR, M1::leg5, M2::leg5)
    @tensor result[a,b,c,d] := ACd[i,j,k,l] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * FL[a,e,f,i]
    return result
end

ACdmap(ACd, FL, FR, M::leg5) = ACdmap(ACd, FL, FR, M, conj(M))
ACdmap(ACd, FL, FR, M::Tuple{leg5,leg5}) = ACdmap(ACd, FL, FR, M[1], M[2])

function Mmap(AC, ACd, FL, FR)
    @tensor result[d,g,e,b] := AC[a,b,c] * FR[c,e,h] * FL[a,d,f] * ACd[f,g,h]
    return result
end
function Mumap(AC, ACd, FL, FR, Mu)
    @tensor result[f,k,h,c,p] := AC[a,b,c,d] * FR[d,g,h,l] * FL[a,e,f,i] * ACd[i,j,k,l] * Mu[e,j,g,b,p]
    return result
end
function Mdmap(AC, ACd, FL, FR, Md)
    @tensor result[e,j,g,b,p] := AC[a,b,c,d] * FR[d,g,h,l] * FL[a,e,f,i] * ACd[i,j,k,l] * Md[f,k,h,c,p]
    return result
end
