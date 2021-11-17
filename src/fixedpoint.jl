using Base.Iterators: drop, take
using IterTools: iterated, imap

"""
    fixedpoint(f, guess, stopfun)

return the result of applying `guess = f(guess)`
until convergence. Convergence is decided by applying
`stopfun(guess)` which returns a Boolean.
"""
function fixedpoint(f, guess, stopfun)
    for state in iterated(f, guess)
        stopfun(state) && return state
    end
end

mutable struct StopFunction{T,S}
    olderror::T
    counter::Int
    tol::S
    maxit::Int
    minit::Int
end

"""
    (st::StopFunction)(state)
    
stopfunction for vumps, returning true if error is smaller than expectation or the maximum
number of iterations is reached. Implemented as a closure since it needs to remember
the last error it saw for comparison.
"""
function (st::StopFunction)(state)
    st.counter += 1
    st.counter > st.maxit - 1 && return true

    error = state[2]
    error <= st.tol && st.counter > st.minit - 1 && return true
    st.olderror = error

    return false
end
