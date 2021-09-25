using AutoHOOT
using ChainRulesCore

const go = AutoHOOT.graphops

# Used to cache the tensor network expressions
# net_sums: An array of NetworkSum. Each element is an array of AutoHOOT nodes (einsum expressions)
# index_dict: A dictionary mapping each AutoHOOT node to the index of the tensor in the input array.
struct NetworkCache
  net_sums::Array{<:NetworkSum}
  index_dict::Dict
end

NetworkCache() = NetworkCache([NetworkSum([])], Dict())

function NetworkCache(networks::Vector{<:Vector{<:AbstractTensor}})
  nodes, node_dict = generate_einsum_expr(networks; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  node_index_dict = generate_node_index_dict(node_dict, networks)
  return NetworkCache(net_sums, node_index_dict)
end

@non_differentiable NetworkCache(networks::Vector{<:Vector{<:AbstractTensor}})

function NetworkCache(trees::Vector{SubNetwork})
  nodes, node_dict = generate_einsum_expr(trees; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  node_index_dict = generate_node_index_dict(node_dict, get_leaves(trees))
  return NetworkCache(net_sums, node_index_dict)
end

@non_differentiable NetworkCache(trees::Vector{SubNetwork})

"""
Generate a cached network, under the constraint that tensors in contract_order will be
contracted based on the order from Array start to the end.
"""
# NOTE: this function is experimental and it hasn't been used in experiments yet.
function NetworkCache(
  networks::Vector{<:Vector{<:AbstractTensor}}, contract_order::Vector{<:AbstractTensor}
)
  nodes, node_dict = generate_einsum_expr(networks)
  constrained_nodes = [retrieve_key(node_dict, t) for t in contract_order]
  function contraction_path(node)
    constrained_inputs = [n for n in constrained_nodes if n in node.inputs]
    out = go.generate_optimal_tree_w_constraint(node, constrained_inputs)
    return out
  end
  net_sums = [NetworkSum([contraction_path(n)]) for n in nodes]
  node_index_dict = generate_node_index_dict(node_dict, networks)
  return NetworkCache(net_sums, node_index_dict)
end

@non_differentiable NetworkCache(
  networks::Vector{<:Vector{<:AbstractTensor}}, contract_order::Vector{<:AbstractTensor}
)
