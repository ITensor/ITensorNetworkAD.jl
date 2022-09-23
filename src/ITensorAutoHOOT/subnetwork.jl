struct SubNetwork
  inputs::Vector{Union{SubNetwork,AbstractTensor}}
end

SubNetwork(inputs::Union{SubNetwork,AbstractTensor}...) = SubNetwork(collect(inputs))

function get_leaf_nodes(trees::Vector{SubNetwork})
  return [get_leaf_nodes(tree) for tree in trees]
end

function get_leaf_nodes(tree::SubNetwork)
  return mapreduce(get_leaf_nodes, vcat, tree.inputs)
end

get_leaf_nodes(tensor::AbstractTensor) = tensor

ITensors.inds(node::SubNetwork) = noncommoninds(get_leaf_nodes(node)...)

# Returns a vector of noncommon indices
ITensors.noncommoninds(node::Union{AbstractTensor,SubNetwork}) = collect(inds(node))

function ITensors.noncommoninds(nodes::Union{AbstractTensor,SubNetwork}...)
  return symdiff(map(inds, nodes)...)
end

function neighboring_tensors(subnetwork::SubNetwork, tensor_list::Vector{<:AbstractTensor})
  subnet_inds = ITensors.inds(subnetwork)
  is_neighbor(t) = length(intersect(subnet_inds, inds(t))) > 0
  return [t for t in tensor_list if is_neighbor(t)]
end

@non_differentiable SubNetwork(inputs::Union{SubNetwork,AbstractTensor}...)
