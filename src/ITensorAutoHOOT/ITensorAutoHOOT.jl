module ITensorAutoHOOT

using ITensors, ChainRulesCore, AutoHOOT

export generate_optimal_tree, gradients
# util functions
export generate_einsum_expr, generate_network, extract_network, compute_graph, retrieve_key
# zygote functions
export batch_tensor_contraction, NetworkCache
# subnetwork
export SubNetwork, get_leaf_nodes, neighboring_tensors
# abstractTensor
export AbstractNetwork, abstract_network

abstract type AbstractNetwork end

AbstractTensor = Union{ITensor,AbstractNetwork}

include("subnetwork.jl")
include("abstractTensor.jl")
include("contraction_AD.jl")
include("network_sum.jl")
include("network_cache.jl")
include("executor.jl")
include("batch_contraction.jl")

end
