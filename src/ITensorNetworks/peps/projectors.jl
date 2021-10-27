using .Models

using ..ITensorAutoHOOT
using ..ITensorAutoHOOT: SubNetwork, neighboring_tensors

function insert_projectors(peps::PEPS, cutoff=1e-15, maxdim=100)
  psi_bra = addtags(linkinds, dag.(peps.data), "bra")
  psi_ket = addtags(linkinds, peps.data, "ket")
  tn = psi_bra .* psi_ket
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  psi_bra_rot = addtags(linkinds, dag.(peps.data), "brarot")
  psi_ket_rot = addtags(linkinds, peps.data, "ketrot")
  tn_rot = psi_bra_rot .* psi_ket_rot
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  bmps_rot = boundary_mps(tn_rot; cutoff=cutoff, maxdim=maxdim)
  # get the projector for each center
  Ny, Nx = size(peps.data)
  bonds_row = [(i, :) for i in 1:Ny]
  bonds_column = [(:, i) for i in 1:Nx]
  tn_split_row, tn_split_column = [], []
  projectors_row, projectors_column = Vector{Vector{ITensor}}(), Vector{Vector{ITensor}}()
  for bond in bonds_row
    tn_split, pl, pr = insert_projectors(tn, bmps; center=bond)
    push!(tn_split_row, tn_split)
    push!(projectors_row, vcat(reduce(vcat, pl), reduce(vcat, pr)))
  end
  for bond in bonds_column
    tn_split, pl, pr = insert_projectors(tn_rot, bmps_rot; center=bond)
    push!(tn_split_column, tn_split)
    push!(projectors_column, vcat(reduce(vcat, pl), reduce(vcat, pr)))
  end
  return tn_split_row, tn_split_column, projectors_row, projectors_column
end

function insert_projectors(peps::PEPS, center::Tuple, cutoff=1e-15, maxdim=100)
  # Square the tensor network
  psi_bra = addtags(linkinds, dag.(peps.data), "bra")
  psi_ket = addtags(linkinds, peps.data, "ket")
  tn = psi_bra .* psi_ket
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  tn_split, pl, pr = insert_projectors(tn, bmps; center=center)
  return tn_split, vcat(reduce(vcat, pl), reduce(vcat, pr))
end

function inner_network(peps::PEPS, peps_prime::PEPS, projectors::Vector{ITensor})
  network = inner_network(peps::PEPS, peps_prime::PEPS)
  return vcat(network, projectors)
end

function inner_network(
  peps::PEPS, peps_prime::PEPS, projectors::Vector{<:ITensor}, ::typeof(tree_w_projectors)
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    line_tensors = vcat(peps.data[i, :], peps_prime.data[i, :])
    neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
    return tree_w_projectors(peps.data[i, :], peps_prime.data[i, :], neighbor_projectors)
  end
  subnetworks = [get_tree(i) for i in 1:Ny]
  return SubNetwork(subnetworks)
end

function inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{<:ITensor},
  mpo::MPO,
  coordinate::Tuple{<:Integer,Colon},
  ::typeof(tree_w_projectors),
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    if i == coordinate[1]
      line_tensors = vcat(peps.data[i, :], peps_prime_ham.data[i, :], mpo.data)
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree_w_projectors(
        peps.data[i, :], peps_prime_ham.data[i, :], mpo.data, neighbor_projectors
      )
    else
      line_tensors = vcat(peps.data[i, :], peps_prime.data[i, :])
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree_w_projectors(peps.data[i, :], peps_prime.data[i, :], neighbor_projectors)
    end
  end
  subnetworks = [get_tree(i) for i in 1:Ny]
  return SubNetwork(subnetworks)
end

function inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{<:ITensor},
  mpo::MPO,
  coordinate::Tuple{Colon,<:Integer},
  ::typeof(tree_w_projectors),
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    if i == coordinate[2]
      line_tensors = vcat(peps.data[:, i], peps_prime_ham.data[:, i], mpo.data)
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree_w_projectors(
        peps.data[:, i], peps_prime_ham.data[:, i], mpo.data, neighbor_projectors
      )
    else
      line_tensors = vcat(peps.data[:, i], peps_prime.data[:, i])
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree_w_projectors(peps.data[:, i], peps_prime.data[:, i], neighbor_projectors)
    end
  end
  subnetworks = [get_tree(i) for i in 1:Nx]
  return SubNetwork(subnetworks)
end

function inner_networks(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, projectors::Vector{ITensor}, Hs::Array
)
  network_list = inner_networks(peps, peps_prime, peps_prime_ham, Hs)
  return map(network -> vcat(network, projectors), network_list)
end

function inner_networks(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{Vector{ITensor}},
  Hs::Vector{Models.LineMPO},
)
  @assert length(projectors) == length(Hs)
  function generate_each_network(projector, H)
    return inner_networks(peps, peps_prime, peps_prime_ham, projector, [H])[1]
  end
  return [generate_each_network(projector, H) for (projector, H) in zip(projectors, Hs)]
end

function inner_networks(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{Vector{ITensor}},
  Hs::Vector{Models.LineMPO},
  ::typeof(tree_w_projectors),
)
  @assert length(projectors) == length(Hs)
  function generate_each_network(projector, H)
    return inner_network(
      peps, peps_prime, peps_prime_ham, projector, H.mpo, H.coord, tree_w_projectors
    )
  end
  return [generate_each_network(projector, H) for (projector, H) in zip(projectors, Hs)]
end
