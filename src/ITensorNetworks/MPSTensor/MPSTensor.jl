using ChainRulesCore

struct MPSTensor
  mps::MPS
end

AbstractTensor = Union{ITensor,MPSTensor}

function ITensors.inds(tensor::MPSTensor)
  return siteinds(tensor.mps) == [nothing] ? () : noncommoninds(tensor.mps...)
end

function Base.getindex(t::MPSTensor)
  @assert length(t.mps) == 1
  @assert order(t.mps[1]) == 0
  return t.mps[1][]
end

function ChainRulesCore.rrule(::typeof(getindex), x::MPSTensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = MPSTensor(MPS([ITensor(ȳ)]))
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

include("contract/contract.jl")
