struct ContractNode
  inputs::Array{Union{ContractNode,ITensor},1}
end

function get_leaves(trees::Array{ContractNode,1})
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
