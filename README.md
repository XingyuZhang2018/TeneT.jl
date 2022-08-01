# VUMPS.jl

<!-- [![Build Status](https://travis-ci.com/XingyuZhang2018/VUMPS.jl.svg?branch=master)](https://travis-ci.com/XingyuZhang2018/VUMPS.jl)
[![Coverage](https://codecov.io/gh/XingyuZhang2018/VUMPS.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/XingyuZhang2018/VUMPS.jl) -->

This is a Julia package for the Variational Uniform Matrix product states(VUMPS) to contract infinite two-dimension square lattice tensor network.

In this package we implemented the VUMPS algorithms including the following:
- Complex number forward and backward propagation
- NixNj Big Unit Cell
- U1-symmmetry and Z2-symmmetry

This package a updated version of the original [ADVUMPS.jl](https://github.com/XingyuZhang2018/ADVUMPS.jl) package, so for more easily to learn and use, we recommend you to use ADVUMPS.jl package.


## install
```shell
> git clone https://github.com/XingyuZhang2018/VUMPS.jl
```
move to the file and run `julia REPL`, press `]` into `Pkg REPL`
```julia
(@v1.7) pkg> activate .
Activating environment at `..\VUMPS\Project.toml`

(VUMPS) pkg> instantiate
```
To get back to the Julia REPL, press `backspace` or `ctrl+C`. Then Precompile `VUMPS`
```julia
julia> using VUMPS
[ Info: Precompiling VUMPS [260a78e0-cbf2-49ba-8157-48058c700f32]
```
## Examples
### 2D Classical Ising Model
A simple example is presentated in `./example/ising.jl` to constructing the tensor for the tensor network representation of the 2d classical Ising Model. 
```julia
julia> include("./example/ising.jl")
```
You can modify Ni and Nj to change the Unit cell size and atype `Array` or `CuArray` to change computation type.

### More Project including this package

- [AD-Kitaev](https://github.com/XingyuZhang2018/AD-Kitaev) - Optimize iPEPS of Kitaev-like model with 1x2 unit cell VUMPS.
- [ADFPEPS.jl](https://github.com/XingyuZhang2018/ADFPEPS.jl) - Optimize fermionic iPEPS of Hubbard model with 2x2 unit cell VUMPS.