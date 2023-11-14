abstract type AbstractSiteType end
export AbstractSiteType
export indextoqn

"""
    indextoqn(::Val{:electronPn}, i::Int)

index to quantum number for electron with party number conservation

state    qn | remainder 
|0 >  -> 0  | 0
|↑ >  -> 1  | 1
|↓ >  -> 1  | 2
|↑↓>  -> 2  | 3

"""
struct electronPn <: AbstractSiteType 
    ifZ2::Bool
end
electronPn() = electronPn(false)
export electronPn
function indextoqn(::electronPn, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 4
        if remainder in [2,3]
            remainder -= 1
        end
        pushfirst!(qni, remainder)
        i = div(i, 4)
    end

    return sum(qni)
end

"""
    indextoqn(i::Int, ::Val{:electronSz})

index to quantum number for electron with spin conservation

state    qn | remainder
|0 >  ->  0 | 0
|↑ >  ->  1 | 1
|↓ >  -> -1 | 2   
|↑↓>  ->  0 | 3

"""
struct electronSz <: AbstractSiteType 
    ifZ2::Bool
end
electronSz() = electronSz(false)
export electronSz
function indextoqn(::electronSz, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 4
        if remainder == 2
            remainder = -1
        elseif remainder == 3
            remainder = 0
        end
        pushfirst!(qni, remainder)
        i = div(i, 4)
    end

    return sum(qni)
end

"""
    indextoqn(i::Int, ::Val{:electronZ2})

index to quantum number for electron Z2 symmetry

state    qn | remainder
|0 >  ->  0 | 0
|↑ >  ->  1 | 1
|↓ >  ->  1 | 2   
|↑↓>  ->  0 | 3

"""
struct electronZ2 <: AbstractSiteType     
    ifZ2::Bool
end
electronZ2() = electronZ2(true)
export electronZ2
function indextoqn(::electronZ2, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 4
        if remainder == 2
            remainder = 1
        elseif remainder == 3
            remainder = 0
        end
        pushfirst!(qni, remainder)
        i = div(i, 4)
    end

    return sum(qni) % 2
end


"""
    indextoqn(i::Int, ::Val{:tJ})

index to quantum number for tJ model

state    qn  | remainder 
|0>  ->  0   | 0
|↑>  ->  1   | 1
|↓>  -> -1   | 2   
"""
struct tJSz <: AbstractSiteType 
    ifZ2::Bool
end
tJSz() = tJSz(false)
export tJSz
function indextoqn(::tJSz, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 3
        remainder == 2 && (remainder = -1)
        pushfirst!(qni, remainder)
        i = div(i, 3)
    end

    return sum(qni)
end

"""
    indextoqn(::Val{:tJZ2}, i::Int)

index to quantum number for tJ Z2 sysmmetry 

state    qn  | remainder 
|0>  ->  0   | 0
|↑>  ->  1   | 1
|↓>  ->  1   | 2   
"""
struct tJZ2 <: AbstractSiteType 
    ifZ2::Bool
end
tJZ2() = tJZ2(true)
export tJZ2
function indextoqn(::tJZ2, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 3
        remainder == 2 && (remainder = 1)
        pushfirst!(qni, remainder)
        i = div(i, 3)
    end

    return sum(qni) % 2
end