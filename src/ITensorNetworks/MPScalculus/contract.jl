using ChainRulesCore

struct MPSTensor
  mps::MPS
end

AbstractTensor = Union{ITensor,MPSTensor}

function ITensors.contract(mps1::MPS, mps2::MPS; cutoff, maxdim)
  ## TODO: modify this function based on https://arxiv.org/pdf/1912.03014.pdf
  tensor = contract(vcat(collect(mps1), collect(mps2))...)
  return if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  end
end

function ITensors.inds(tensor::MPSTensor)
  return siteinds(tensor.mps) == [nothing] ? [] : siteinds(tensor.mps)
end

# contract into one ITensor
ITensors.ITensor(t::MPSTensor) = contract(collect(t.mps)...)

ITensors.contract(t1::MPSTensor; kwargs...) = t1

function ITensors.contract(t1::MPSTensor, t2::MPSTensor; kwargs...)
  mps_out = contract(t1.mps, t2.mps; kwargs...)
  return MPSTensor(mps_out)
end

function ITensors.contract(t1::MPSTensor, t2::MPSTensor...; kwargs...)
  return contract(t1, contract(t2...; kwargs...); kwargs...)
end

ITensors.contract(t_list::Vector{MPSTensor}; kwargs...) = contract(t_list...; kwargs...)

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
