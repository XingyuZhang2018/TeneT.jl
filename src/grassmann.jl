"""
    Project a vector g onto a Grassmann (ij)(k) manifold at point x
    x: shape: (i,j)(k)
    g: shape: (i,j)(k)
"""
project_AL(∂AL, AL) = project_AL!(deepcopy(∂AL), AL)
function project_AL!(∂AL, AL)
    # ∂AL .-= ein"deg,(abc,abg)->dec"(AL, ∂AL, conj.(AL))
    @tensoropt out[d,e,c] := AL[a,b,c] * ∂AL[a,b,g] * conj(AL[d,e,g])
    ∂AL .-= out
    return ∂AL
end

project_AR(∂AR, AR) = project_AR!(deepcopy(∂AR), AR)
function project_AR!(∂AR, AR)
    ∂AL = permute_fronttail(∂AR)
    AL = permute_fronttail(AR)
    ∂AR .= permute_fronttail(project_AL!(∂AL, AL))
    return ∂AR
end

"""
    Retrct a vector g onto a Grassmann manifold at point x.
     This is a SVD based retraction and same as it in the Stiefel manifold.
"""
function retract!(x)
    AL, _, _ = left_canonical(x)
    x .= AL
    return x
end