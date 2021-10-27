using Random
using .Models

"""
A finite size PEPS type.
"""
struct PEPS
  data::Matrix{ITensor}
end

PEPS(Nx::Int, Ny::Int) = PEPS(Matrix{ITensor}(undef, Nx, Ny))

"""
    PEPS([::Type{ElT} = Float64, sites; linkdims=1)
Construct an PEPS filled with Empty ITensors of type `ElT` from a collection of indices.
Optionally specify the link dimension with the keyword argument `linkdims`, which by default is 1.
"""
function PEPS(::Type{T}, sites::Matrix{<:Index}; linkdims::Integer=1) where {T<:Number}
  Ny, Nx = size(sites)
  tensor_grid = Matrix{ITensor}(undef, Ny, Nx)
  # we assume the PEPS at least has size (2,2). Can generalize if necessary
  @assert(Nx >= 2 && Ny >= 2)

  lh = Matrix{Index}(undef, Ny, Nx - 1)
  for ii in 1:(Nx - 1)
    for jj in 1:(Ny)
      lh[jj, ii] = Index(linkdims, "Lh-$jj-$ii")
    end
  end
  lv = Matrix{Index}(undef, Ny - 1, Nx)
  for ii in 1:(Nx)
    for jj in 1:(Ny - 1)
      lv[jj, ii] = Index(linkdims, "Lv-$jj-$ii")
    end
  end

  # boundary cases
  tensor_grid[1, 1] = ITensor(T, lh[1, 1], lv[1, 1], sites[1, 1])
  tensor_grid[1, Nx] = ITensor(T, lh[1, Nx - 1], lv[1, Nx], sites[1, Nx])
  tensor_grid[Ny, 1] = ITensor(T, lh[Ny, 1], lv[Ny - 1, 1], sites[Ny, 1])
  tensor_grid[Ny, Nx] = ITensor(T, lh[Ny, Nx - 1], lv[Ny - 1, Nx], sites[Ny, Nx])
  for ii in 2:(Nx - 1)
    tensor_grid[1, ii] = ITensor(T, lh[1, ii], lh[1, ii - 1], lv[1, ii], sites[1, ii])
    tensor_grid[Ny, ii] = ITensor(
      T, lh[Ny, ii], lh[Ny, ii - 1], lv[Ny - 1, ii], sites[Ny, ii]
    )
  end

  # inner sites
  for jj in 2:(Ny - 1)
    tensor_grid[jj, 1] = ITensor(T, lh[jj, 1], lv[jj, 1], lv[jj - 1, 1], sites[jj, 1])
    tensor_grid[jj, Nx] = ITensor(
      T, lh[jj, Nx - 1], lv[jj, Nx], lv[jj - 1, Nx], sites[jj, Nx]
    )
    for ii in 2:(Nx - 1)
      tensor_grid[jj, ii] = ITensor(
        T, lh[jj, ii], lh[jj, ii - 1], lv[jj, ii], lv[jj - 1, ii], sites[jj, ii]
      )
    end
  end

  return PEPS(tensor_grid)
end

PEPS(sites::Matrix{<:Index}, args...; kwargs...) = PEPS(Float64, sites, args...; kwargs...)

function Random.randn!(P::PEPS)
  randn!.(P.data)
  normalize!.(P.data)
  return P
end

Base.:+(A::PEPS, B::PEPS) = broadcast_add(A, B)

broadcast_add(A::PEPS, B::PEPS) = PEPS(A.data .+ B.data)

broadcast_minus(A::PEPS, B::PEPS) = PEPS(A.data .- B.data)

broadcast_mul(c::Number, A::PEPS) = PEPS(c .* A.data)

broadcast_inner(A::PEPS, B::PEPS) = mapreduce(v -> v[], +, A.data .* B.data)

ITensors.prime(P::PEPS, n::Integer=1) = PEPS(map(x -> prime(x, n), P.data))

# prime a PEPS with specified indices
function ITensors.prime(indices::Array{<:Index,1}, P::PEPS, n::Integer=1)
  return PEPS(prime(indices, P.data, n))
end

# prime linkinds of a PEPS
function ITensors.prime(::typeof(linkinds), P::PEPS, n::Integer=1)
  return PEPS(mapinds(x -> prime(x, n), linkinds, P.data))
end

function ITensors.addtags(::typeof(linkinds), P::PEPS, args...)
  return PEPS(addtags(linkinds, P.data, args...))
end

function ITensors.removetags(::typeof(linkinds), P::PEPS, args...)
  return PEPS(removetags(linkinds, P.data, args...))
end

ITensors.data(P::PEPS) = P.data

split_network(P::PEPS, rotation=false) = PEPS(split_network(data(P); rotation=rotation))

function ITensors.commoninds(p1::PEPS, p2::PEPS)
  return commoninds(p1.data, p2.data)
end

function flatten(v::Array{<:PEPS})
  tensor_list = [vcat(peps.data...) for peps in v]
  return vcat(tensor_list...)
end

function rayleigh_quotient(inners::Array)
  self_inner = inners[end][]
  expectations = sum(inners[1:(end - 1)])[]
  return expectations / self_inner
end

include("tree.jl")
include("inner_networks.jl")
include("projectors.jl")
include("chain_rules.jl")
