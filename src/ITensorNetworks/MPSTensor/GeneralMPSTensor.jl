using ChainRulesCore
using ZygoteRules: @adjoint

using ..ITensorAutoHOOT
using ..ITensorAutoHOOT: AbstractNetwork

struct GeneralMPSTensor <: AbstractNetwork
  mps::MPS
end

function Base.:+(A::GeneralMPSTensor, B::GeneralMPSTensor; kwargs...)
  if length(A.mps) == 1 && order(A.mps[1]) == 0
    return GeneralMPSTensor(MPS([ITensor(A[] + B[])]))
  else
    return GeneralMPSTensor(+(A.mps, B.mps; kwargs...))
  end
end

function Base.getindex(t::GeneralMPSTensor)
  @assert length(t.mps) == 1
  @assert order(t.mps[1]) == 0
  return t.mps[1][]
end

function ChainRulesCore.rrule(::typeof(getindex), x::GeneralMPSTensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = GeneralMPSTensor(MPS([ITensor(ȳ)]))
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

"""
initialization methods
"""
# TODO: general tags are not comparable
Base.isless(a::Index, b::Index) = id(a) < id(b) || (id(a) == id(b) && plev(a) < plev(b)) # && tags(a) < tags(b)

function GeneralMPSTensor(tensor::ITensor; cutoff, maxdim)
  mps_out = if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, sort(inds(tensor)); cutoff=cutoff, maxdim=maxdim)
  end
  return GeneralMPSTensor(mps_out)
end

@adjoint function GeneralMPSTensor(tensor::ITensor; cutoff, maxdim)
  MPSTensor_pullback(dtensor) = (ITensor(dtensor),)
  return GeneralMPSTensor(tensor; cutoff=cutoff, maxdim=maxdim), MPSTensor_pullback
end

"""
contract
"""
function general_mps_contract(mps1::MPS, mps2::MPS; cutoff, maxdim)
  ## TODO
  tensor = contract(vcat(collect(mps1), collect(mps2))...)
  return if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, sort(inds(tensor)); cutoff=cutoff, maxdim=maxdim)
  end
end

function ITensors.contract(t1::GeneralMPSTensor, t2::GeneralMPSTensor; cutoff, maxdim)
  mps_out = general_mps_contract(t1.mps, t2.mps; cutoff=cutoff, maxdim=maxdim)
  return GeneralMPSTensor(mps_out)
end
