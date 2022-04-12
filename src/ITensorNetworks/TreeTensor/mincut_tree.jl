using Graphs, GraphsFlows, Combinatorics
using ..Profiler

# a large number to prevent this edge being a cut
MAX_WEIGHT = 100000

@profile function graph_generation(network::Vector{ITensor}, uncontract_inds::Vector)
  edge_dict = Dict()
  # only go over contracted inds
  contract_edges = []
  for (i, t) in enumerate(network)
    for ind in setdiff(inds(t), uncontract_inds)
      if !haskey(edge_dict, ind)
        edge_dict[ind] = (i, log2(space(ind)))
      else
        @assert(length(edge_dict[ind]) == 2)
        edge_dict[ind] = (edge_dict[ind][1], i, edge_dict[ind][2])
        push!(contract_edges, edge_dict[ind])
      end
    end
  end
  graph = Graphs.DiGraph(length(network))
  capacity_matrix = zeros(length(network), length(network))
  for e in contract_edges
    u, v, f = e
    Graphs.add_edge!(graph, u, v)
    Graphs.add_edge!(graph, v, u)
    capacity_matrix[u, v] = f
    capacity_matrix[v, u] = f
  end
  # update uncontract inds
  grouped_uncontracted_inds = []
  for (i, t) in enumerate(network)
    ucinds = intersect(inds(t), uncontract_inds)
    if length(ucinds) == 0
      continue
    end
    for ind in ucinds
      push!(grouped_uncontracted_inds, [ind])
      edge_dict[[ind]] = (i, log2(space(ind)))
    end
  end
  return graph, capacity_matrix, edge_dict, grouped_uncontracted_inds
end

@profile function inds_binary_tree(
  network::Vector{ITensor},
  uncontract_inds::Vector;
  algorithm="mincut",
  groupinds_tree=nothing,
)
  if algorithm == "sequential-mps"
    out_inds = [uncontract_inds[1]]
    for i in 2:length(uncontract_inds)
      out_inds = [out_inds, [uncontract_inds[i]]]
    end
    return out_inds
  end
  graph, capacity_matrix, edge_dict, grouped_uncontracted_inds = graph_generation(
    network, uncontract_inds
  )
  if algorithm == "mincut"
    return mincut_inds(graph, capacity_matrix, edge_dict, grouped_uncontracted_inds)
  elseif algorithm == "mincut-mps"
    inds_tree = mincut_inds(graph, capacity_matrix, edge_dict, grouped_uncontracted_inds)
    linear_tree = linearize(inds_tree, graph, capacity_matrix, edge_dict)
    out_inds = linear_tree[1]
    for i in 2:length(linear_tree)
      out_inds = [out_inds, linear_tree[i]]
    end
    return out_inds
  elseif algorithm == "mps"
    return mps_inds(graph, capacity_matrix, edge_dict, grouped_uncontracted_inds)
  end
end

function linearize(
  inds_tree::Vector, graph::Graphs.DiGraph, capacity_matrix::Matrix, edge_dict::Dict
)
  get_dist(edge, distances) = distances.dists[edge_dict[edge][1]]
  function get_boundary_dists(line, source)
    first, last = line[1], line[end]
    ds = dijkstra_shortest_paths(graph, source, capacity_matrix)
    return get_dist(first, ds), get_dist(last, ds)
  end

  if length(inds_tree) == 1
    return inds_tree
  end
  left = linearize(inds_tree[1], graph, capacity_matrix, edge_dict)
  right = linearize(inds_tree[2], graph, capacity_matrix, edge_dict)
  if length(left) == 1 && length(right) == 1
    return [left, right]
  end
  if length(left) == 1
    source = edge_dict[left][1]
    dist_first, dist_last = get_boundary_dists(right, source)
    if dist_last < dist_first
      right = reverse(right)
    end
    return [left, right...]
  end
  if length(right) == 1
    source = edge_dict[right][1]
    dist_first, dist_last = get_boundary_dists(left, source)
    if dist_last > dist_first
      left = reverse(left)
    end
    return [left..., right]
  end
  s1, s2 = edge_dict[left[1]][1], edge_dict[left[end]][1]
  dist1_first, dist1_last = get_boundary_dists(right, s1)
  dist2_first, dist2_last = get_boundary_dists(right, s2)
  if min(dist1_first, dist1_last) < min(dist2_first, dist2_last)
    left = reverse(left)
    if dist1_last < dist1_first
      right = reverse(right)
    end
  else
    if dist2_last < dist2_first
      right = reverse(right)
    end
  end
  return [left..., right...]
end

@profile function mincut_subnetwork(
  network::Vector{ITensor}, sourceinds::Vector, uncontract_inds::Vector
)
  if length(sourceinds) == length(uncontract_inds)
    return network
  end
  graph, capacity_matrix, edge_dict, grouped_uncontracted_inds = graph_generation(
    network, uncontract_inds
  )
  grouped_sourceinds = [[ind] for ind in sourceinds]
  part1, part2, mincut = mincut_value(
    graph, capacity_matrix, edge_dict, grouped_uncontracted_inds, grouped_sourceinds
  )
  @assert length(part1) > 1
  return [network[i] for i in part1 if i <= length(network)]
end

@profile function mincut_inds(
  graph::Graphs.DiGraph, capacity_matrix::Matrix, edge_dict::Dict, uncontract_inds::Vector
)
  graph = copy(graph)
  capacity_matrix = copy(capacity_matrix)
  edge_dict = copy(edge_dict)
  uncontract_inds = copy(uncontract_inds)
  # base case here
  if length(uncontract_inds) <= 2
    return uncontract_inds
  end
  new_edge, minval = new_edge_mincut(graph, capacity_matrix, edge_dict, uncontract_inds)
  new_capacity_matrix, uncontract_inds = update!(
    graph, capacity_matrix, edge_dict, uncontract_inds, new_edge, minval
  )
  return mincut_inds(graph, new_capacity_matrix, edge_dict, uncontract_inds)
end

function mps_inds(
  graph::Graphs.DiGraph, capacity_matrix::Matrix, edge_dict::Dict, uncontract_inds::Vector
)
  # base case here
  if length(uncontract_inds) <= 2
    return uncontract_inds
  end
  new_edge, minval = new_edge_mincut(graph, capacity_matrix, edge_dict, uncontract_inds, 1)
  first_ind = new_edge[1]
  s = edge_dict[first_ind][1]
  ds = dijkstra_shortest_paths(graph, s, capacity_matrix)
  get_dist(edge) = ds.dists[edge_dict[edge][1]]
  remain_inds = [i for i in uncontract_inds if i != first_ind]
  sort!(remain_inds; by=get_dist)
  out_inds = first_ind
  for i in remain_inds
    out_inds = [out_inds, i]
  end
  return out_inds
end

# update the graph
function update!(
  graph::Graphs.DiGraph,
  capacity_matrix::Matrix,
  edge_dict::Dict,
  uncontract_inds::Vector,
  new_edge::Vector,
  minval,
)
  add_vertex!(graph)
  last_vertex = size(graph)[1]
  u1, w_u1 = edge_dict[new_edge[1]]
  u2, w_u2 = edge_dict[new_edge[2]]
  Graphs.add_edge!(graph, u1, last_vertex)
  Graphs.add_edge!(graph, u2, last_vertex)
  Graphs.add_edge!(graph, last_vertex, u1)
  Graphs.add_edge!(graph, last_vertex, u2)
  new_capacity_matrix = zeros(last_vertex, last_vertex)
  new_capacity_matrix[1:(last_vertex - 1), 1:(last_vertex - 1)] = capacity_matrix
  new_capacity_matrix[u1, last_vertex] = MAX_WEIGHT
  new_capacity_matrix[u2, last_vertex] = MAX_WEIGHT
  new_capacity_matrix[last_vertex, u1] = MAX_WEIGHT
  new_capacity_matrix[last_vertex, u2] = MAX_WEIGHT
  # update the dict
  edge_dict[new_edge[1]] = (u1, last_vertex, MAX_WEIGHT)#w_u1)
  edge_dict[new_edge[2]] = (u2, last_vertex, MAX_WEIGHT)#w_u2)
  edge_dict[new_edge] = (last_vertex, minval)
  # update uncontract_inds
  uncontract_inds = setdiff(uncontract_inds, new_edge)
  uncontract_inds = vcat([new_edge], uncontract_inds)
  return new_capacity_matrix, uncontract_inds
end

function new_edge_mincut(
  graph::Graphs.DiGraph,
  capacity_matrix::Matrix,
  edge_dict::Dict,
  uncontract_inds::Vector,
  size=2,
)
  split_inds_list = collect(powerset(uncontract_inds, size, size))
  mincuts = [
    mincut_value(graph, capacity_matrix, edge_dict, uncontract_inds, split_inds)[3] for
    split_inds in split_inds_list
  ]
  split_sizes = [
    sum([edge_dict[ind][2] for ind in split_inds]) for split_inds in split_inds_list
  ]
  weights = [min(mincuts[i], split_sizes[i]) for i in 1:length(mincuts)]
  indices_min = [i for i in 1:length(mincuts) if weights[i] == min(weights...)]
  cuts_min = [mincuts[i] for i in indices_min]
  _, index = findmin(cuts_min)
  i = indices_min[index]
  minval = weights[i]
  new_edge = split_inds_list[i]
  return new_edge, minval
end

function mincut_value(
  graph::Graphs.DiGraph,
  capacity_matrix::Matrix,
  edge_dict::Dict,
  uncontract_inds::Vector,
  split_inds::Vector,
)
  graph = copy(graph)
  # add two vertices to the graph to model the s and t
  add_vertices!(graph, 2)
  t = size(graph)[1]
  s = t - 1
  new_capacity_matrix = zeros(t, t)
  new_capacity_matrix[1:(t - 2), 1:(t - 2)] = capacity_matrix
  for ind in split_inds
    u, _ = edge_dict[ind]
    Graphs.add_edge!(graph, u, s)
    Graphs.add_edge!(graph, s, u)
    new_capacity_matrix[u, s] = MAX_WEIGHT
    new_capacity_matrix[s, u] = MAX_WEIGHT
  end
  terminal_inds = setdiff(uncontract_inds, split_inds)
  for ind in terminal_inds
    u, _ = edge_dict[ind]
    Graphs.add_edge!(graph, u, t)
    Graphs.add_edge!(graph, t, u)
    new_capacity_matrix[u, t] = MAX_WEIGHT
    new_capacity_matrix[t, u] = MAX_WEIGHT
  end
  # this t and s sequence makes sure part1 is the largest subgraph yielding mincut
  part2, part1, flow = GraphsFlows.mincut(
    graph, t, s, new_capacity_matrix, EdmondsKarpAlgorithm()
  )
  return part1, part2, flow
end
