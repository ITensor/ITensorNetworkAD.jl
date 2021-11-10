"""
Transfer ITensor networks into networks with type tensortype
"""
function abstract_network!(tensortype, network::Vector{ITensor}, dict::Dict; kwargs...)
  for tensor in network
    if !haskey(dict, tensor)
      dict[tensor] = tensortype(tensor; kwargs...)
    end
  end
  return Vector{tensortype}([dict[t] for t in network])
end

function abstract_network!(tensortype, tree::SubNetwork, dict::Dict; kwargs...)
  for node in tree.inputs
    if !haskey(dict, node)
      if node isa ITensor
        dict[node] = tensortype(node; kwargs...)
      else
        dict[node] = abstract_network!(tensortype, node, dict; kwargs...)
      end
    end
  end
  return SubNetwork([dict[n] for n in tree.inputs])
end

function abstract_network(tensortype, networks::Vector{Vector{ITensor}}, vars; kwargs...)
  dict = Dict{ITensor,tensortype}()
  if length(vars) != 0
    vars = abstract_network!(tensortype, collect(vars), dict; kwargs...)
  end
  networks = Vector{Vector{tensortype}}([
    abstract_network!(tensortype, n, dict; kwargs...) for n in networks
  ])
  return networks, Tuple(vars)
end

function ChainRulesCore.rrule(
  ::typeof(abstract_network), tensortype, networks::Vector{Vector{ITensor}}, vars; kwargs...
)
  function pullback(v)
    d_networks, dvars = v[1], v[2]
    dvars_itensor = map(x -> ITensor(x), dvars)
    return (NoTangent(), NoTangent(), NoTangent(), dvars_itensor)
  end
  return abstract_network(tensortype, networks, vars; kwargs...), pullback
end

function abstract_network(tensortype, trees::Vector{SubNetwork}, vars; kwargs...)
  dict = Dict()
  if length(vars) != 0
    vars = abstract_network!(tensortype, collect(vars), dict; kwargs...)
  end
  trees = Vector{SubNetwork}([
    abstract_network!(tensortype, t, dict; kwargs...) for t in trees
  ])
  return trees, Tuple(vars)
end

function ChainRulesCore.rrule(
  ::typeof(abstract_network), tensortype, trees::Vector{SubNetwork}, vars; kwargs...
)
  function pullback(v)
    d_trees, dvars = v[1], v[2]
    dvars_itensor = map(x -> ITensor(x), dvars)
    return (NoTangent(), NoTangent(), NoTangent(), dvars_itensor)
  end
  return abstract_network(tensortype, trees, vars; kwargs...), pullback
end
