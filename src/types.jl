# Core abstract types and type aliases for TeneT.jl

# Lattice types — embedded in HamiltonianModel structs
abstract type AbstractLattice end
struct Square <: AbstractLattice end
struct Honeycomb <: AbstractLattice end
struct KagomeLattice <: AbstractLattice end

# Contraction modes for VUMPS specialization
abstract type ContractionMode end
struct General <: ContractionMode end
struct Plaquette <: ContractionMode end

# Boundary algorithm base type
abstract type Algorithm end

# iPEPS optimization base type
abstract type iPEPSOptimize end

# Hamiltonian model base type
abstract type HamiltonianModel end
