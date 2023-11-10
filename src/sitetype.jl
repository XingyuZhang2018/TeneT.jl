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
struct electronPn <: AbstractSiteType end
export electronPn
function indextoqn(::electronPn, i::Int)
    i -= 1
    i == 0 && return 0

    ternary = []
    while i > 0
        remainder = i % 4
        if remainder in [2,3]
            remainder -= 1
        end
        pushfirst!(ternary, remainder)
        i = div(i, 4)
    end

    return sum(ternary)
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
struct electronSz <: AbstractSiteType end
export electronSz
function indextoqn(::electronSz, i::Int)
    i -= 1
    i == 0 && return 0

    ternary = []
    while i > 0
        remainder = i % 4
        if remainder == 2
            remainder = -1
        elseif remainder == 3
            remainder = 0
        end
        pushfirst!(ternary, remainder)
        i = div(i, 4)
    end

    return sum(ternary)
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
struct electronZ2 <: AbstractSiteType end
export electronZ2
function indextoqn(::electronZ2, i::Int)
    i -= 1
    i == 0 && return 0

    ternary = []
    while i > 0
        remainder = i % 4
        if remainder == 2
            remainder = 1
        elseif remainder == 3
            remainder = 0
        end
        pushfirst!(ternary, remainder)
        i = div(i, 4)
    end

    return sum(ternary) % 2
end


"""
    indextoqn(i::Int, ::Val{:tJ})

index to quantum number for tJ model

state    qn  | remainder 
|0>  ->  0   | 0
|↑>  ->  1   | 1
|↓>  -> -1   | 2   
"""
struct tJSz <: AbstractSiteType end
export tJSz
function indextoqn(::tJSz, i::Int)
    i -= 1
    i == 0 && return 0

    ternary = []
    while i > 0
        remainder = i % 3
        remainder == 2 && (remainder = -1)
        pushfirst!(ternary, remainder)
        i = div(i, 3)
    end

    return sum(ternary)
end

"""
    indextoqn(::Val{:tJZ2}, i::Int)

index to quantum number for tJ Z2 sysmmetry 

state    qn  | remainder 
|0>  ->  0   | 0
|↑>  ->  1   | 1
|↓>  ->  1   | 2   
"""
struct tJZ2 <: AbstractSiteType end
export tJZ2
function indextoqn(::tJZ2, i::Int)
    i -= 1
    i == 0 && return 0

    ternary = []
    while i > 0
        remainder = i % 3
        remainder == 2 && (remainder = 1)
        pushfirst!(ternary, remainder)
        i = div(i, 3)
    end

    return sum(ternary) % 2
end