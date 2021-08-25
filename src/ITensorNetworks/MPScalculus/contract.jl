using ChainRulesCore

struct MPSTensor
  mps::MPS
  cutoff::Float64
  maxdim::Integer
end

AbstractTensor = Union{ITensor,MPSTensor}

function ITensors.contract(mps1::MPS, mps2::MPS; cutoff=1e-15, maxdim=1000)
  ## TODO: modify this function based on https://arxiv.org/pdf/1912.03014.pdf
  tensor = contract(vcat(collect(mps1), collect(mps2))...)
  return if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  end
end

MPSTensor(mps::MPS) = MPSTensor(mps, 1e-15, 1000)

function ITensors.inds(tensor::MPSTensor)
  return siteinds(tensor.mps) == [nothing] ? [] : siteinds(tensor.mps)
end

# contract into one ITensor
ITensors.ITensor(t::MPSTensor) = contract(collect(t.mps)...)

ITensors.contract(t1::MPSTensor) = t1

function ITensors.contract(t1::MPSTensor, t2::MPSTensor)
  mps_out = contract(t1.mps, t2.mps; cutoff=t1.cutoff, maxdim=t1.maxdim)
  return MPSTensor(mps_out, t1.cutoff, t1.maxdim)
end

ITensors.contract(t1::MPSTensor, t2::MPSTensor...) = contract(t1, contract(t2...))

ITensors.contract(t_list::Vector{MPSTensor}) = contract(t_list...)

function Base.getindex(t::MPSTensor)
  @assert length(t.mps) == 1
  @assert order(t.mps[1]) == 0
  return t.mps[1][]
end

function ChainRulesCore.rrule(::typeof(getindex), x::MPSTensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = MPSTensor(MPS([ITensor(ȳ)]), x.cutoff, x.maxdim)
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end
