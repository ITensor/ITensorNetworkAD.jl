using ChainRulesCore

struct MPSTensor
  mps::MPS
end

function ITensors.inds(tensor::MPSTensor)
  return siteinds(tensor.mps) == [nothing] ? () : noncommoninds(tensor.mps...)
end

function Base.:+(A::MPSTensor, B::MPSTensor; kwargs...)
  if length(A.mps) == 1 && order(A.mps[1]) == 0
    return MPSTensor(MPS([ITensor(A[] + B[])]))
  else
    return MPSTensor(+(A.mps, B.mps; kwargs...))
  end
end

ITensors.sum(tensors::Vector{MPSTensor}; kwargs...) = ITensors.sum(tensors...; kwargs...)

function ITensors.sum(t1::MPSTensor, t2::MPSTensor...; kwargs...)
  return +(t1, sum(t2...; kwargs...); kwargs...)
end

ITensors.sum(t::MPSTensor; kwargs...) = t

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

include("methods/general_mps.jl")
include("methods/singleindex_mps.jl")

"""
initialization methods
"""
function MPSTensor(tensor::ITensor; cutoff, maxdim, method)
  if method == "general_mps"
    return general_mps_MPSTensor(tensor; cutoff=cutoff, maxdim=maxdim)
  elseif method == "singleindex_mps"
    return singleindex_mps_MPSTensor(tensor; cutoff=cutoff, maxdim=maxdim)
  end
end

function ChainRulesCore.rrule(::typeof(MPSTensor), tensor::ITensor; cutoff, maxdim, method)
  MPSTensor_pullback(dtensor) = (NoTangent(), ITensor(dtensor))
  return MPSTensor(tensor; cutoff=cutoff, maxdim=maxdim, method=method), MPSTensor_pullback
end

"""
contract
"""
# contract into one ITensor
ITensors.ITensor(t::MPSTensor; kwargs...) = contract(collect(t.mps)...; kwargs...)

function ITensors.contract(t1::MPSTensor, t2::MPSTensor; cutoff, maxdim, method)
  if method == "general_mps"
    return general_mps_contract(t1, t2; cutoff=cutoff, maxdim=maxdim)
  elseif method == "singleindex_mps"
    return singleindex_mps_contract(t1, t2; cutoff=cutoff, maxdim=maxdim)
  end
end

ITensors.contract(t1::MPSTensor; kwargs...) = t1

function ITensors.contract(t1::MPSTensor, t2::MPSTensor...; kwargs...)
  return contract(t1, contract(t2...; kwargs...); kwargs...)
end

ITensors.contract(t_list::Vector{MPSTensor}; kwargs...) = contract(t_list...; kwargs...)
