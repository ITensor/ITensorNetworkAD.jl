using ChainRulesCore

include("SimpleMPSTensor.jl")
include("GeneralMPSTensor.jl")

MPSTensor = Union{GeneralMPSTensor,SimpleMPSTensor}

function ITensors.inds(tensor::MPSTensor)
  return siteinds(tensor.mps) == [nothing] ? () : noncommoninds(tensor.mps...)
end

ITensors.sum(tensors::Vector{<:MPSTensor}; kwargs...) = ITensors.sum(tensors...; kwargs...)

function ITensors.sum(t1::MPSTensor, t2::MPSTensor...; kwargs...)
  return +(t1, sum(t2...; kwargs...); kwargs...)
end

ITensors.sum(t::MPSTensor; kwargs...) = t

"""
contract
"""
# contract into one ITensor
ITensors.ITensor(t::MPSTensor; kwargs...) = contract(collect(t.mps)...; kwargs...)

ITensors.contract(t1::MPSTensor; kwargs...) = t1

function ITensors.contract(t1::MPSTensor, t2::MPSTensor...; kwargs...)
  return contract(t1, contract(t2...; kwargs...); kwargs...)
end

ITensors.contract(t_list::Vector{<:MPSTensor}; kwargs...) = contract(t_list...; kwargs...)
