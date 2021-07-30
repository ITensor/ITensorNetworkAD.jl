struct SubNetwork
  inputs::Vector{Union{SubNetwork,ITensor}}
end

SubNetwork(inputs::Union{SubNetwork,ITensor}...) = SubNetwork(collect(inputs))

function get_leaves(trees::Vector{SubNetwork})
  return [get_leaves(tree) for tree in trees]
end

function get_leaves(tree::SubNetwork)
  return mapreduce(get_leaves, vcat, tree.inputs)
end

get_leaves(tensor::ITensor) = tensor

ITensors.inds(node::SubNetwork) = noncommoninds(get_leaves(node)...)

function ITensors.noncommoninds(nodes::Union{ITensor,SubNetwork}...)
  return symdiff(map(inds, nodes)...)
end
