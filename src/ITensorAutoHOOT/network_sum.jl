
# represent a sum of tensor networks
struct NetworkSum
  nodes::Array
end

# TODO: add caching intermediates here
function run(net_sum::NetworkSum, feed_dict::Dict)
  return ITensors.sum(compute_graph(net_sum.nodes, feed_dict))
end

inner(net_sum::NetworkSum, n2) = NetworkSum([inner(n1, n2) for n1 in net_sum.nodes])

Base.push!(net_sum::NetworkSum, node) = Base.push!(net_sum.nodes, node)
