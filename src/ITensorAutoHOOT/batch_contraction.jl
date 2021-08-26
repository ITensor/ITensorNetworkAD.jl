using AutoHOOT
using ChainRulesCore

using ..ITensorNetworks
using ..ITensorNetworks: SubNetwork, AbstractTensor

const ad = AutoHOOT.autodiff

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

batch_tensor_contraction(executor::Executor, vars...; kwargs...) = run(executor; kwargs...)

function ChainRulesCore.rrule(
  ::typeof(batch_tensor_contraction), executor::Executor, vars...; kwargs...
)
  function pullback(v)
    output = batch_tensor_contraction(vjps(executor, vars, v), vars...; kwargs...)
    return (NoTangent(), NoTangent(), Tuple(output)...)
  end
  return batch_tensor_contraction(executor, vars...; kwargs...), pullback
end

"""Perform a batch of tensor contractions, each one defined by a tensor network.
Parameters
----------
networks: An array of networks. Each network is represented by an array of tensors
vars: the tensors to take derivative of
Returns
-------
A list of tensors representing the contraction outputs of each network.
"""
function batch_tensor_contraction(
  networks::Vector{<:Vector{<:AbstractTensor}}, vars...; kwargs...
)
  return batch_tensor_contraction(Executor(networks), vars...; kwargs...)
end

function batch_tensor_contraction(trees::Vector{SubNetwork}, vars...; kwargs...)
  return batch_tensor_contraction(Executor(trees), vars...; kwargs...)
end

function batch_tensor_contraction(
  networks::Vector{<:Vector{<:AbstractTensor}}, cache::NetworkCache, vars...; kwargs...
)
  return batch_tensor_contraction(Executor(networks, cache), vars...; kwargs...)
end

function batch_tensor_contraction(
  trees::Vector{SubNetwork}, cache::NetworkCache, vars...; kwargs...
)
  return batch_tensor_contraction(Executor(trees, cache), vars...; kwargs...)
end
