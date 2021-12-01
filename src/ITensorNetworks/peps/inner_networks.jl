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
  network = SubNetwork(peps.data[1, :]..., peps_prime.data[1, :]...)
  for i in 2:Ny
    network = SubNetwork(network, peps.data[i, :]..., peps_prime.data[i, :]...)
  end
  return network
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
  network1, network2 = SubNetwork(), SubNetwork()
  if coordinate[1] > 1
    network1 = SubNetwork(peps.data[1, :]..., peps_prime.data[1, :]...)
    for i in 2:(coordinate[1] - 1)
      network1 = SubNetwork(network1, peps.data[i, :]..., peps_prime.data[i, :]...)
    end
  end
  if coordinate[1] < Ny
    network2 = SubNetwork(peps.data[Ny, :]..., peps_prime.data[Ny, :]...)
    for i in reverse((coordinate[1] + 1):(Ny - 1))
      network2 = SubNetwork(network2, peps.data[i, :]..., peps_prime.data[i, :]...)
    end
  end
  if coordinate[1] == 1
    return SubNetwork(
      network2, peps.data[1, :]..., peps_prime_ham.data[1, :]..., mpo.data...
    )
  elseif coordinate[1] == Ny
    return SubNetwork(
      network1, peps.data[Ny, :]..., peps_prime_ham.data[Ny, :]..., mpo.data...
    )
  else
    index = coordinate[1]
    return SubNetwork(
      network1,
      network2,
      peps.data[index, :]...,
      peps_prime_ham.data[index, :]...,
      mpo.data...,
    )
  end
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
  network1, network2 = SubNetwork(), SubNetwork()
  if coordinate[2] > 1
    network1 = SubNetwork(peps.data[:, 1]..., peps_prime.data[:, 1]...)
    for i in 2:(coordinate[2] - 1)
      network1 = SubNetwork(network1, peps.data[:, i]..., peps_prime.data[:, i]...)
    end
  end
  if coordinate[2] < Nx
    network2 = SubNetwork(peps.data[:, Nx]..., peps_prime.data[:, Nx]...)
    for i in reverse((coordinate[2] + 1):(Nx - 1))
      network2 = SubNetwork(network2, peps.data[:, i]..., peps_prime.data[:, i]...)
    end
  end
  if coordinate[2] == 1
    return SubNetwork(
      network2, peps.data[:, 1]..., peps_prime_ham.data[:, 1]..., mpo.data...
    )
  elseif coordinate[2] == Nx
    return SubNetwork(
      network1, peps.data[:, Nx]..., peps_prime_ham.data[:, Nx]..., mpo.data...
    )
  else
    index = coordinate[2]
    return SubNetwork(
      network1,
      network2,
      peps.data[:, index]...,
      peps_prime_ham.data[:, index]...,
      mpo.data...,
    )
  end
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
