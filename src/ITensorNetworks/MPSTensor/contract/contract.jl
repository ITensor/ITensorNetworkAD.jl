include("general_mps.jl")
include("singleindex_mps.jl")

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
