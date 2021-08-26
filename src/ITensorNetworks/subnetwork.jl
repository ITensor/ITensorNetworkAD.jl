using ChainRulesCore

struct SubNetwork
  inputs::Vector{Union{SubNetwork,AbstractTensor}}
end

SubNetwork(inputs::Union{SubNetwork,AbstractTensor}...) = SubNetwork(collect(inputs))

function get_leaves(trees::Vector{SubNetwork})
  return [get_leaves(tree) for tree in trees]
end

function get_leaves(tree::SubNetwork)
  return mapreduce(get_leaves, vcat, tree.inputs)
end

get_leaves(tensor::AbstractTensor) = tensor

ITensors.inds(node::SubNetwork) = noncommoninds(get_leaves(node)...)

function ITensors.noncommoninds(nodes::Union{AbstractTensor,SubNetwork}...)
  return symdiff(map(inds, nodes)...)
end

function neighboring_tensors(subnetwork::SubNetwork, tensor_list::Vector{<:AbstractTensor})
  subnet_inds = ITensors.inds(subnetwork)
  is_neighbor(t) = length(intersect(subnet_inds, inds(t))) > 0
  return [t for t in tensor_list if is_neighbor(t)]
end

@non_differentiable SubNetwork(inputs::Union{SubNetwork,AbstractTensor}...)
