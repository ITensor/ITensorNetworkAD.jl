const ad = AutoHOOT.autodiff
const go = AutoHOOT.graphops

include("utils.jl")

# Calculate the gradient of the output scalar represented by
# a network w.r.t. the input tensors.
function gradients(network::Array, in_tensors::Array)
  nodes, dict = generate_einsum_expr([network])
  in_nodes = [retrieve_key(dict, t) for t in in_tensors]
  node = go.generate_optimal_tree(nodes[1])
  grads = ad.gradients(node, in_nodes)
  return [extract_network(g, dict) for g in grads]
end

function generate_optimal_tree(network::Array)
  nodes, dict = generate_einsum_expr([network])
  node = go.generate_optimal_tree(nodes[1])
  return extract_network(node, dict)
end
