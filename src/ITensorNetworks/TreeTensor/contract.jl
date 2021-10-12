
# contract into one ITensor
ITensors.ITensor(t::TreeTensor; kwargs...) = contract(collect(t.tensors)...; kwargs...)

ITensors.contract(t1::TreeTensor; kwargs...) = t1

function ITensors.contract(t1::TreeTensor, t2::TreeTensor...; kwargs...)
  return contract(t1, contract(t2...; kwargs...); kwargs...)
end

ITensors.contract(t_list::Vector{TreeTensor}; kwargs...) = contract(t_list...; kwargs...)

function ITensors.contract(t1::TreeTensor, t2::TreeTensor; cutoff, maxdim)
  connect_inds = intersect(inds(t1), inds(t2))
  if length(connect_inds) <= 1
    return TreeTensor(t1.tensors..., t2.tensors...)
  else
    # TODO: currently it only transfer the tree to ITensor, do the contraction, then trasferring the output to an MPS
    out = contract(t1.tensors..., t2.tensors...)
    if size(out) == ()
      return TreeTensor(out)
    else
      mps = MPS(out, inds(out); cutoff=cutoff, maxdim=maxdim)
      return TreeTensor(mps...)
    end
  end
end
