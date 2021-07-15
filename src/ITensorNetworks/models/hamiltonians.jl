using ITensors

struct LocalMPO
  mpo::MPO
  coord1::Pair{<:Integer,<:Integer}
  coord2::Pair{<:Integer,<:Integer}
end

# Transverse field
# The critical point is h = 1.0
# This is the most challenging part of the model for DMRG
function mpo(::Model"tfim", sites::Matrix{<:Index}; h::Float64)
  Ny, Nx = size(sites)
  sites_vec = reshape(sites, Nx * Ny)
  lattice = square_lattice(Nx, Ny; yperiodic=false)

  opsum = OpSum()
  for b in lattice
    opsum += -1, "X", b.s1, "X", b.s2
  end
  for i in 1:(Nx * Ny)
    opsum += h, "Z", i
  end
  return MPO(opsum, sites_vec)
end

# Get the local hamiltonian term of a 2D grid
function localham_term(::Model"tfim", sites::Matrix{<:Index}, bond; h::Float64)
  Ny, Nx = size(sites)
  sites_vec = reshape(sites, Nx * Ny)
  n1, n2 = bond.s1, bond.s2
  opsum = OpSum()
  opsum += -1, "X", 1, "X", 2
  if n2 == n1 + 1
    opsum += h, "Z", 1
  end
  if n2 == n1 + 1 && n2 % Ny == 0
    opsum += h, "Z", 2
  end
  mpo = MPO(opsum, sites_vec[[n1, n2]])
  coord1 = ((n1 - 1) % Ny + 1) => trunc(Int, (n1 - 1) / Ny) + 1
  coord2 = ((n2 - 1) % Ny + 1) => trunc(Int, (n2 - 1) / Ny) + 1
  return LocalMPO(mpo, coord1, coord2)
end

# Return a list of LocalMPO
function localham(m::Model, sites; kwargs...)
  Ny, Nx = size(sites)
  lattice = square_lattice(Nx, Ny; yperiodic=false)
  return [localham_term(m, sites, bond; kwargs...) for bond in lattice]
end

# Check that the local Hamiltonian is the same as the MPO
function checklocalham(Hlocal, H, sites)
  @disable_warn_order begin
    Ny, Nx = size(sites)
    sites_vec = reshape(sites, Nx * Ny)
    lattice = square_lattice(Nx, Ny; yperiodic=false)

    # This scales exponentially
    Hlocal_full = ITensor()
    for (i, bond) in enumerate(lattice)
      Hlocalterm_full = prod(Hlocal[i].mpo)
      n1, n2 = bond.s1, bond.s2
      for m in 1:(Nx * Ny)
        if !(m in (n1, n2))
          Hlocalterm_full *= op("Id", sites_vec, m)
        end
      end
      Hlocal_full += Hlocalterm_full
    end
    @show norm(Hlocal_full - prod(H))
  end
  return isapprox(norm(Hlocal_full), norm(prod(H)))
end
