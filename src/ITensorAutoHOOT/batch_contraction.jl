using AutoHOOT
using ChainRulesCore

using ..ITensorNetworks
using ..ITensorNetworks: SubNetwork, get_leaves

const ad = AutoHOOT.autodiff
const go = AutoHOOT.graphops

# represent a sum of tensor networks
struct NetworkSum
  nodes::Array
end

struct Executor
  net_sums::Array{<:NetworkSum}
  feed_dict::Dict
end

# Used to cache the tensor network expressions
# net_sums: An array of NetworkSum. Each element is an array of AutoHOOT nodes (einsum expressions)
# index_dict: A dictionary mapping each AutoHOOT node to the index of the tensor in the input array.
struct NetworkCache
  net_sums::Array{<:NetworkSum}
  index_dict::Dict
end

NetworkSum() = NetworkSum([])

function Executor(networks::Vector{Vector{ITensor}})
  nodes, node_dict = generate_einsum_expr(networks; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  return Executor(net_sums, node_dict)
end

@non_differentiable Executor(networks::Vector{Vector{ITensor}})

function Executor(trees::Vector{SubNetwork})
  nodes, node_dict = generate_einsum_expr(trees; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  return Executor(net_sums, node_dict)
end

@non_differentiable Executor(trees::Vector{SubNetwork})

function Executor(networks::Vector{Vector{ITensor}}, cache::NetworkCache)
  node_dict = Dict()
  for (node, index) in cache.index_dict
    i, j = index
    node_dict[node] = networks[i][j]
  end
  return Executor(cache.net_sums, node_dict)
end

@non_differentiable Executor(networks::Vector{Vector{ITensor}}, cache::NetworkCache)

function Executor(trees::Vector{SubNetwork}, cache::NetworkCache)
  return Executor(get_leaves(trees), cache)
end

@non_differentiable Executor(trees::Vector{SubNetwork}, cache::NetworkCache)

NetworkCache() = NetworkCache([NetworkSum([])], Dict())

function NetworkCache(networks::Vector{Vector{ITensor}})
  nodes, node_dict = generate_einsum_expr(networks; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  node_index_dict = generate_node_index_dict(node_dict, networks)
  return NetworkCache(net_sums, node_index_dict)
end

@non_differentiable NetworkCache(networks::Vector{Vector{ITensor}})

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
function NetworkCache(networks::Vector{Vector{ITensor}}, contract_order::Vector{ITensor})
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
  networks::Vector{Vector{ITensor}}, contract_order::Vector{ITensor}
)

# TODO: add caching intermediates here
function run(net_sum::NetworkSum, feed_dict::Dict)
  return ITensors.sum(compute_graph(net_sum.nodes, feed_dict))
end

inner(net_sum::NetworkSum, n2) = NetworkSum([inner(n1, n2) for n1 in net_sum.nodes])

Base.push!(net_sum::NetworkSum, node) = Base.push!(net_sum.nodes, node)

function run(executor::Executor)
  return [run(net_sum, executor.feed_dict) for net_sum in executor.net_sums]
end

Base.length(executor::Executor) = Base.length(executor.net_sums)

function construct_gradient!(net_sum::NetworkSum, innodes::Array, feed_dict::Dict)
  for net in net_sum.nodes
    inputs = ad.get_all_inputs(net)
    vars = [n for n in inputs if n in innodes]
    grads = ad.gradients(net, vars)
    for (grad, var) in zip(grads, vars)
      push!(feed_dict[var], grad)
    end
  end
end

# vector-jacobian product
function vjps(executor::Executor, vars, vector)
  node_dict = copy(executor.feed_dict)
  for t in vector
    update_dict!(node_dict, t)
  end
  # add vector to the executor
  @assert(length(vector) == length(executor))
  vec_nodes = [retrieve_key(node_dict, t) for t in vector]
  net_sums = [
    inner(net_sum, vnode) for (net_sum, vnode) in zip(executor.net_sums, vec_nodes)
  ]
  # get the gradient graph
  innodes = [retrieve_key(node_dict, t) for t in vars]
  network_sum_dict = Dict()
  for n in innodes
    network_sum_dict[n] = NetworkSum()
  end
  for net_sum in net_sums
    construct_gradient!(net_sum, innodes, network_sum_dict)
  end
  return Executor([network_sum_dict[n] for n in innodes], node_dict)
end

@non_differentiable vjps(executor::Executor, vars, vector)

batch_tensor_contraction(executor::Executor, vars...) = run(executor)

function ChainRulesCore.rrule(
  ::typeof(batch_tensor_contraction), executor::Executor, vars...
)
  function pullback(v)
    output = batch_tensor_contraction(vjps(executor, vars, v), vars...)
    return (NoTangent(), NoTangent(), Tuple(output)...)
  end
  return batch_tensor_contraction(executor, vars...), pullback
end

"""Perform a batch of tensor contractions, each one defined by a tensor network.
Parameters
----------
networks: An array of networks. Each network is represented by an array of ITensor tensors
vars: the tensors to take derivative of
Returns
-------
A list of tensors representing the contraction outputs of each network.
"""
function batch_tensor_contraction(networks::Vector{Vector{ITensor}}, vars...)
  return batch_tensor_contraction(Executor(networks), vars...)
end

function batch_tensor_contraction(trees::Vector{SubNetwork}, vars...)
  return batch_tensor_contraction(Executor(trees), vars...)
end

function batch_tensor_contraction(
  networks::Vector{Vector{ITensor}}, cache::NetworkCache, vars...
)
  return batch_tensor_contraction(Executor(networks, cache), vars...)
end

function batch_tensor_contraction(trees::Vector{SubNetwork}, cache::NetworkCache, vars...)
  return batch_tensor_contraction(Executor(trees, cache), vars...)
end
