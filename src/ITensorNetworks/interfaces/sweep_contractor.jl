using SweepContractor
using ..ITensorAutoHOOT
using SweepContractor: TensorNetwork, LabelledTensorNetwork
using ..ITensorAutoHOOT: SubNetwork

function ITensor_networks(TN::TensorNetwork)
  SweepContractor.sort!(TN)
  index_dict = Dict()
  function itensor(i, t)
    # Construct indices
    inds = []
    for (dim, j) in enumerate(t.adj)
      label = sort([i, j])
      if !haskey(index_dict, label)
        s = size(t.arr)[dim]
        index_dict[label] = Index(s, string(label))
      end
      push!(inds, index_dict[label])
    end
    # build the tensor
    return ITensor(t.arr, inds...)
  end
  inetwork = [itensor(i, t) for (i, t) in enumerate(TN)]
  return inetwork
end

function ITensor_networks(LTN::LabelledTensorNetwork)
  return ITensor_networks(SweepContractor.delabel(LTN))
end

function line_network(network::Vector)
  if length(network) <= 2
    return SubNetwork(network)
  end
  return SubNetwork(line_network(network[1:(end - 1)]), network[end])
end
