using ChainRulesCore

struct Executor
  net_sums::Array{<:NetworkSum}
  feed_dict::Dict
end

NetworkSum() = NetworkSum([])

function Executor(networks::Vector{<:Vector{<:AbstractTensor}})
  nodes, node_dict = generate_einsum_expr(networks; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  return Executor(net_sums, node_dict)
end

@non_differentiable Executor(networks::Vector{<:Vector{<:AbstractTensor}})

function Executor(trees::Vector{SubNetwork})
  nodes, node_dict = generate_einsum_expr(trees; optimize=true)
  net_sums = [NetworkSum([n]) for n in nodes]
  return Executor(net_sums, node_dict)
end

@non_differentiable Executor(trees::Vector{SubNetwork})

function Executor(networks::Vector{<:Vector{<:AbstractTensor}}, cache::NetworkCache)
  node_dict = Dict()
  for (node, index) in cache.index_dict
    i, j = index
    node_dict[node] = networks[i][j]
  end
  return Executor(cache.net_sums, node_dict)
end

@non_differentiable Executor(
  networks::Vector{<:Vector{<:AbstractTensor}}, cache::NetworkCache
)

function Executor(trees::Vector{SubNetwork}, cache::NetworkCache)
  return Executor(get_leaves(trees), cache)
end

@non_differentiable Executor(trees::Vector{SubNetwork}, cache::NetworkCache)

function run(executor::Executor; kwargs...)
  return [run(net_sum, executor.feed_dict; kwargs...) for net_sum in executor.net_sums]
end

Base.length(executor::Executor) = Base.length(executor.net_sums)
