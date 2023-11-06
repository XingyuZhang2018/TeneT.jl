"""
    index_to_parity(n::Int, ::Val{:electron})

for electron, we can use bits to label the Z2 parity

state         bit | parity
|0 >  = 00 ->  0  |   0
|↑ >  = 01 ->  1  |   1
|↓ >  = 10 ->  1  |   1
|↑↓>  = 11 ->  2  |   0

"""
function index_to_parity(n::Int, ::Val{:electron})
    n -= 1
    n == 0 && return 0

    ternary = []
    while n > 0
        remainder = n % 2
        pushfirst!(ternary, remainder)
        n = div(n, 2)
    end

    return sum(ternary) % 2
end

"""
    index_to_parity(n::Int, ::Val{:tJ})

for tJ, we can use triplets to label the Z2 parity

state    triplet | parity
|0>  =      0    |   0
|↑>  =      1    |   1
|↓>  =      2    |   1
"""
function index_to_parity(n::Int, ::Val{:tJ})
    n -= 1
    n == 0 && return 0

    ternary = []
    while n > 0
        remainder = n % 3
        remainder == 2 && (remainder = 1)
        pushfirst!(ternary, remainder)
        n = div(n, 3)
    end

    return sum(ternary) % 2
end