struct ContractNode
  inputs::Vector{Union{ContractNode,ITensor}}
end

ContractNode(inputs::Union{ContractNode,ITensor}...) = ContractNode(collect(inputs))

function get_leaves(trees::Vector{ContractNode})
  return [get_leaves(tree) for tree in trees]
end

function get_leaves(tree::ContractNode)
  return mapreduce(get_leaves, vcat, tree.inputs)
end

get_leaves(tensor::ITensor) = tensor

ITensors.inds(node::ContractNode) = noncommoninds(get_leaves(node)...)

function ITensors.noncommoninds(nodes::Union{ITensor,ContractNode}...)
  return symdiff(map(inds, nodes)...)
end
