using ChainRulesCore

struct MPSTensor
  mps::Union{MPS,ITensor}
  cutoff::Float64
  maxdim::Integer
end

AbstractTensor = Union{ITensor,MPSTensor}

function ITensors.contract(mps1::MPS, mps2::MPS; cutoff=1e-15, maxdim=1000)
  ## TODO: modify this function based on https://arxiv.org/pdf/1912.03014.pdf
  tensor = contract(vcat(mps1.data, mps2.data)...)
  out =
    size(tensor) == () ? tensor : MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  return out
end

function ITensors.contract(mps1::MPS, mps2::ITensor; cutoff=1e-15, maxdim=1000)
  tensor = contract(vcat(mps1.data, [mps2])...)
  out =
    size(tensor) == () ? tensor : MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  return out
end

function ITensors.contract(mps1::ITensor, mps2::MPS; cutoff=1e-15, maxdim=1000)
  return contract(mps2, mps1; cutoff=cutoff, maxdim=maxdim)
end

MPSTensor(mps::MPS) = MPSTensor(mps, 1e-15, 1000)

function ITensors.inds(tensor::MPSTensor)
  return tensor.mps isa MPS ? siteinds(tensor.mps) : inds(tensor.mps)
end

# contract into one ITensor
function ITensors.ITensor(t::MPSTensor)
  return t.mps isa MPS ? contract(t.mps.data) : t.mps
end

ITensors.contract(t1::MPSTensor) = t1

function ITensors.contract(t1::MPSTensor, t2::MPSTensor)
  mps_out = contract(t1.mps, t2.mps; cutoff=t1.cutoff, maxdim=t1.maxdim)
  return MPSTensor(mps_out, t1.cutoff, t1.maxdim)
end

ITensors.contract(t1::MPSTensor, t2::MPSTensor...) = contract(t1, contract(t2...))

ITensors.contract(t_list::Vector{MPSTensor}) = contract(t_list...)

function Base.getindex(t::MPSTensor)
  @assert t.mps isa ITensor
  @assert size(t.mps) == ()
  return t.mps[]
end

function ChainRulesCore.rrule(::typeof(getindex), x::MPSTensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = MPSTensor(ITensor(ȳ), x.cutoff, x.maxdim)
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end
