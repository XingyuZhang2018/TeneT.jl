# Core abstract types and type aliases for TeneT.jl

# Lattice types — embedded in HamiltonianModel structs
abstract type AbstractLattice end
struct Square <: AbstractLattice end
struct Honeycomb{Mode} <: AbstractLattice end
Honeycomb(s::Symbol) = Honeycomb{s}()
struct Kagome{Mode} <: AbstractLattice end
Kagome(s::Symbol) = Kagome{s}()

# Filesystem-safe string representations (avoid : { } characters for Windows paths)
Base.show(io::IO, ::Square)              = print(io, "Square")
Base.show(io::IO, ::Honeycomb{M}) where M = print(io, "Honeycomb_", M)
Base.show(io::IO, ::Kagome{M}) where M    = print(io, "Kagome_", M)

# Contraction modes for VUMPS specialization
abstract type ContractionMode end
struct General <: ContractionMode end
struct C4v <: ContractionMode end
struct Plaquette{Mode} <: ContractionMode end
Plaquette(L::AbstractLattice) = Plaquette{typeof(L)}()

# Boundary algorithm base type
abstract type Algorithm end

# iPEPS optimization base type
abstract type iPEPSOptimize end

# Hamiltonian model base type
abstract type HamiltonianModel end

# Filesystem-safe show for all HamiltonianModel subtypes
# Produces e.g. "Heisenberg_Square(S=0.5,Jx=-1.0,Jy=-1.0,Jz=1.0,ifrotate=true)" instead of
# "Heisenberg{Square}(Square(), 0.5, -1.0, -1.0, 1.0, true)"
function Base.show(io::IO, model::HamiltonianModel)
    # type name without parameter: Heisenberg{Square} -> Heisenberg
    name = nameof(typeof(model))
    print(io, name, "_", model.lattice, "(")
    first = true
    for f in fieldnames(typeof(model))
        f === :lattice && continue
        f === :bondratio && continue
        v = getfield(model, f)
        first ? (first = false) : print(io, ",")
        print(io, f, "=", v)
    end
    print(io, ")")
end
