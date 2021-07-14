using AutoHOOT
using ChainRulesCore
using ITensors: setinds

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

NetworkSum() = NetworkSum([])

function Executor(networks::Array)
  nodes, node_dict = generate_einsum_expr(networks)
  # TODO: add caching here
  net_sums = [NetworkSum([go.generate_optimal_tree(n)]) for n in nodes]
  return Executor(net_sums, node_dict)
end

@non_differentiable Executor(networks::Array)

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
function batch_tensor_contraction(networks::Array, vars...)
  return batch_tensor_contraction(Executor(networks), vars...)
end
