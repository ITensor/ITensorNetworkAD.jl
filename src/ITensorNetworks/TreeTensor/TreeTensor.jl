using ChainRulesCore
using ZygoteRules: @adjoint

using ..ITensorAutoHOOT
using ..ITensorAutoHOOT: AbstractNetwork

struct TreeTensor <: AbstractNetwork
  tensors::Vector{ITensor}
end

function Base.show(io::IO, tree::TreeTensor)
  out_str = "\n"
  for (i, t) in enumerate(tree.tensors)
    out_str = out_str * "[" * string(i) * "] " * string(inds(t)) * "\n"
  end
  return print(io, out_str)
end

TreeTensor(tensors::ITensor...; kwargs...) = TreeTensor(collect(tensors))

@adjoint function TreeTensor(tensor::ITensor)
  treetensor_pullback(dtensor) = (ITensor(dtensor),)
  return TreeTensor(tensor), treetensor_pullback
end

ITensors.inds(tree::TreeTensor) = tuple(noncommoninds(tree.tensors...)...)

function Base.getindex(t::TreeTensor)
  @assert length(t.tensors) == 1
  @assert order(t.tensors[1]) == 0
  return t.tensors[1][]
end

function ChainRulesCore.rrule(::typeof(getindex), x::TreeTensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = TreeTensor([ITensor(ȳ)])
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

function Base.:+(A::TreeTensor, B::TreeTensor; kwargs...)
  if length(A.tensors) == 1 && length(B.tensors) == 1
    return TreeTensor(A.tensors[1] + B.tensors[1])
  else
    # TODO: currently it only transfer the tree to ITensor, do the addition, then trasferring the output to an MPS
    out = ITensor(A) + ITensor(B)
    mps = MPS(out, inds(out); kwargs...)
    return TreeTensor(mps...)
  end
end

ITensors.sum(tensors::Vector{TreeTensor}; kwargs...) = ITensors.sum(tensors...; kwargs...)

function ITensors.sum(t1::TreeTensor, t2::TreeTensor...; kwargs...)
  return +(t1, sum(t2...; kwargs...); kwargs...)
end

ITensors.sum(t::TreeTensor; kwargs...) = t

include("mincut_tree.jl")
include("contract.jl")
