module ITensorAutoHOOT

export generate_optimal_tree, gradients
# util functions
export generate_einsum_expr, generate_network, extract_network, compute_graph, retrieve_key
# zygote functions
export batch_tensor_contraction, scalar

include("contraction_AD.jl")
include("zygote.jl")

end
