using ChainRulesCore
using ZygoteRules: @adjoint

struct SimpleMPSTensor <: AbstractNetwork
  mps::MPS
end

function Base.:+(A::SimpleMPSTensor, B::SimpleMPSTensor; kwargs...)
  if length(A.mps) == 1 && order(A.mps[1]) == 0
    return SimpleMPSTensor(MPS([ITensor(A[] + B[])]))
  else
    return SimpleMPSTensor(+(A.mps, B.mps; kwargs...))
  end
end

function Base.getindex(t::SimpleMPSTensor)
  @assert length(t.mps) == 1
  @assert order(t.mps[1]) == 0
  return t.mps[1][]
end

function ChainRulesCore.rrule(::typeof(getindex), x::SimpleMPSTensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = SimpleMPSTensor(MPS([ITensor(ȳ)]))
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

"""
initialization methods
"""
function SimpleMPSTensor(tensor::ITensor; cutoff, maxdim)
  mps_out = if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  end
  return SimpleMPSTensor(mps_out)
end

@adjoint function SimpleMPSTensor(tensor::ITensor; cutoff, maxdim)
  MPSTensor_pullback(dtensor) = (ITensor(dtensor),)
  return SimpleMPSTensor(tensor; cutoff=cutoff, maxdim=maxdim), MPSTensor_pullback
end

"""
contract
"""
function simple_mps_contract(mps1::MPS, mps2::MPS; cutoff, maxdim)
  ## TODO: modify this function based on https://arxiv.org/pdf/1912.03014.pdf
  tensor = contract(vcat(collect(mps1), collect(mps2))...)
  return if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  end
end

function ITensors.contract(t1::SimpleMPSTensor, t2::SimpleMPSTensor; cutoff, maxdim)
  mps_out = simple_mps_contract(t1.mps, t2.mps; cutoff=cutoff, maxdim=maxdim)
  return SimpleMPSTensor(mps_out)
end
