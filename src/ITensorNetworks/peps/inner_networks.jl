using .Models

# Get the tensor network of <peps|peps'>
function inner_network(peps::PEPS, peps_prime::PEPS)
  return vcat(vcat(peps.data...), vcat(peps_prime.data...))
end

# Get the tensor network of <peps|mpo|peps'>
# The local MPO specifies the 2-site term of the Hamiltonian
function inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, mpo::MPO, coordinates::Array
)
  @assert(length(mpo) == length(coordinates))
  network = vcat(peps.data...)
  dimy, dimx = size(peps.data)
  for ii in 1:dimx
    for jj in 1:dimy
      if (jj, ii) in coordinates
        index = findall(x -> x == (jj, ii), coordinates)
        @assert(length(index) == 1)
        network = vcat(network, [mpo.data[index[1]]])
        network = vcat(network, [peps_prime_ham.data[jj, ii]])
      else
        network = vcat(network, [peps_prime.data[jj, ii]])
      end
    end
  end
  return network
end

"""Generate an array of networks representing inner products, <p|H_1|p>, ..., <p|H_n|p>
Parameters
----------
peps: a peps network with datatype PEPS
peps_prime: prime of peps used for inner products
peps_prime_ham: prime of peps used for calculating expectation values
Hs: An array of MPO operators with datatype LocalMPO
Returns
-------
An array of networks.
"""
function inner_networks(peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, Hs::Array)
  network_list = Vector{Vector{ITensor}}()
  for H_term in Hs
    if H_term isa Models.LocalMPO
      coords = [H_term.coord1, H_term.coord2]
    elseif H_term isa Models.LineMPO
      if H_term.coord[1] isa Colon
        coords = [(i, H_term.coord[2]) for i in 1:length(H_term.mpo)]
      else
        coords = [(H_term.coord[1], i) for i in 1:length(H_term.mpo)]
      end
    end
    inner = inner_network(peps, peps_prime, peps_prime_ham, H_term.mpo, coords)
    network_list = vcat(network_list, [inner])
  end
  return network_list
end

function inner_network(peps::PEPS, peps_prime::PEPS, ::typeof(tree))
  Ny, Nx = size(peps.data)
  function get_tree(i)
    return tree(peps.data[i, :], peps_prime.data[i, :])
  end
  subnetworks = [get_tree(i) for i in 1:Ny]
  return SubNetwork(subnetworks)
end

function inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  mpo::MPO,
  coordinate::Tuple{<:Integer,Colon},
  ::typeof(tree),
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    if i == coordinate[1]
      return tree(peps.data[i, :], peps_prime_ham.data[i, :], mpo.data)
    else
      return tree(peps.data[i, :], peps_prime.data[i, :])
    end
  end
  subnetworks = [get_tree(i) for i in 1:Ny]
  return SubNetwork(subnetworks)
end

function inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  mpo::MPO,
  coordinate::Tuple{Colon,<:Integer},
  ::typeof(tree),
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    if i == coordinate[2]
      return tree(peps.data[:, i], peps_prime_ham.data[:, i], mpo.data)
    else
      return tree(peps.data[:, i], peps_prime.data[:, i])
    end
  end
  subnetworks = [get_tree(i) for i in 1:Nx]
  return SubNetwork(subnetworks)
end

function inner_networks(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  Hs::Vector{Models.LineMPO},
  ::typeof(tree),
)
  function generate_each_network(H)
    return inner_network(peps, peps_prime, peps_prime_ham, H.mpo, H.coord, tree)
  end
  return [generate_each_network(H) for H in Hs]
end
