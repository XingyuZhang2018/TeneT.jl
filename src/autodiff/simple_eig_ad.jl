# simple_eig AD: Zygote backpropagates through each power iteration step directly.
# No custom rrule needed — simple_eig in utils/misc.jl is written in a
# Zygote-differentiable way (each step is v = f(v) / norm(v)).
#
# The old KrylovKit-based eigsolve pullback has been removed.
# Rationale: for power iteration with a small number of steps,
# direct backprop through each step is simpler and sufficient.
