"""
A finite size PEPS type.
"""
mutable struct PEPS
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
      lh[jj, ii] = Index(linkdims, "Lh,$jj,$ii")
    end
  end
  lv = Matrix{Index}(undef, Ny - 1, Nx)
  for ii in 1:(Nx)
    for jj in 1:(Ny - 1)
      lv[jj, ii] = Index(linkdims, "Lv,$jj,$ii")
    end
  end

  # boundary cases
  tensor_grid[1, 1] = emptyITensor(T, lh[1, 1], lv[1, 1], sites[1, 1])
  tensor_grid[1, Nx] = emptyITensor(T, lh[1, Nx - 1], lv[1, Nx], sites[1, Nx])
  tensor_grid[Ny, 1] = emptyITensor(T, lh[Ny, 1], lv[Ny - 1, 1], sites[Ny, 1])
  tensor_grid[Ny, Nx] = emptyITensor(T, lh[Ny, Nx - 1], lv[Ny - 1, Nx], sites[Ny, Nx])
  for ii in 2:(Nx - 1)
    tensor_grid[1, ii] = emptyITensor(T, lh[1, ii], lh[1, ii - 1], lv[1, ii], sites[1, ii])
    tensor_grid[Ny, ii] = emptyITensor(
      T, lh[Ny, ii], lh[Ny, ii - 1], lv[Ny - 1, ii], sites[Ny, ii]
    )
  end

  # inner sites
  for jj in 2:(Ny - 1)
    tensor_grid[jj, 1] = emptyITensor(T, lh[jj, 1], lv[jj, 1], lv[jj - 1, 1], sites[jj, 1])
    tensor_grid[jj, Nx] = emptyITensor(
      T, lh[jj, Nx - 1], lv[jj, Nx], lv[jj - 1, Nx], sites[jj, Nx]
    )
    for ii in 2:(Nx - 1)
      tensor_grid[jj, ii] = emptyITensor(
        T, lh[jj, ii], lh[jj, ii - 1], lv[jj, ii], lv[jj - 1, ii], sites[jj, ii]
      )
    end
  end

  return PEPS(tensor_grid)
end

PEPS(sites::Matrix{<:Index}, args...; kwargs...) = PEPS(Float64, sites, args...; kwargs...)

PEPS(data::Array, size::Tuple{Int64,Int64}) = PEPS(reshape(data, size[1], size[2]))

function randomizePEPS!(P::PEPS)
  dimy, dimx = size(P.data)
  for ii in 1:dimx
    for jj in 1:dimy
      randn!(P.data[jj, ii])
      normalize!(P.data[jj, ii])
    end
  end
end

function Base.:+(A::PEPS, B::PEPS)
  @assert(size(A.data) == size(B.data))
  out = [t1 + t2 for (t1, t2) in zip(vcat(A.data...), vcat(B.data...))]
  return PEPS(out, size(A.data))
end

function Base.:-(A::PEPS, B::PEPS)
  @assert(size(A.data) == size(B.data))
  out = [t1 - t2 for (t1, t2) in zip(vcat(A.data...), vcat(B.data...))]
  return PEPS(out, size(A.data))
end

function Base.:*(c::Number, A::PEPS)
  out = [c * t for t in vcat(A.data...)]
  return PEPS(out, size(A.data))
end

function Base.:*(A::PEPS, B::PEPS)
  @assert(size(A.data) == size(B.data))
  out_scalar = 0
  for (t1, t2) in zip(vcat(A.data...), vcat(B.data...))
    out_scalar += ITensors.scalar(t1 * t2)
  end
  return out_scalar
end

function ITensors.prime(peps::PEPS; ham=true)
  prime_peps = []
  for tensor in vcat(peps.data...)
    indices = inds(tensor)
    if ham == true
      bonds = indices
    else
      bonds = [indices[i] for i in 1:(length(indices) - 1)]
    end
    prime_peps = vcat(prime_peps, [ITensors.prime(tensor, bonds)])
  end
  return PEPS(prime_peps, size(peps.data))
end

# Get the tensor network of <peps|peps'>
function inner_network(peps::PEPS, peps_prime::PEPS)
  return vcat(vcat(peps.data...), vcat(peps_prime.data...))
end

# Get the tensor network of <peps|mpo|peps'>
# The local MPO specifies the 2-site term of the Hamiltonian
function inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, mpo::MPO, coordinates::Array
)
  @assert(length(mpo) == length(coordinates))
  network = vcat(peps.data...)
  dimy, dimx = size(peps.data)
  for ii in 1:dimx
    for jj in 1:dimy
      if (jj => ii) in coordinates
        index = findall(x -> x == (jj => ii), coordinates)
        @assert(length(index) == 1)
        network = vcat(network, [mpo.data[index[1]]])
        network = vcat(network, [peps_prime_ham.data[jj, ii]])
      else
        network = vcat(network, [peps_prime.data[jj, ii]])
      end
    end
  end
  return network
end

function extract_data(v::Array{<:PEPS})
  tensor_list = [vcat(peps.data...) for peps in v]
  return vcat(tensor_list...)
end
