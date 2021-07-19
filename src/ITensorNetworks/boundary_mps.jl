
#
# Boundary MPS contraction methods
#

#
# Approximately contract a 2D tensor network with boundary MPS.
# The boundary MPS will be used as projectors to be inserted
# into the network.
#

struct ContractionAlgorithm{algorithm} end

ContractionAlgorithm(s::AbstractString) = ContractionAlgorithm{Symbol(s)}()

macro ContractionAlgorithm_str(s)
  return :(ContractionAlgorithm{$(Expr(:quote, Symbol(s)))})
end

Base.@kwdef struct BoundaryMPS
  top::Vector{MPS} = MPS[]
  bottom::Vector{MPS} = MPS[]
  left::Vector{MPS} = MPS[]
  right::Vector{MPS} = MPS[]
end

struct BoundaryMPSDir{dir} end

BoundaryMPSDir(s::AbstractString) = BoundaryMPSDir{Symbol(s)}()

macro BoundaryMPSDir_str(s)
  return :(BoundaryMPSDir{$(Expr(:quote, Symbol(s)))})
end

function contract_approx(
  tn::Matrix{ITensor},
  alg::ContractionAlgorithm"boundary_mps",
  dirs::BoundaryMPSDir"top_to_bottom";
  cutoff,
  maxdim,
)
  nrows, ncols = size(tn)
  boundary_mps = Vector{MPS}(undef, nrows - 1)
  x = MPS(tn[1, :])
  boundary_mps[1] = orthogonalize(x, ncols)
  for nrow in 2:(nrows - 1)
    A = MPO(tn[nrow, :])
    x = contract(A, x; cutoff=cutoff, maxdim=maxdim)
    boundary_mps[nrow] = orthogonalize(x, ncols)
  end
  return BoundaryMPS(; top=boundary_mps)
end

function contract_approx(
  tn::Matrix{ITensor},
  alg::ContractionAlgorithm"boundary_mps",
  dirs::BoundaryMPSDir"bottom_to_top";
  cutoff,
  maxdim,
)
  tn = rot180(tn)
  tn = reverse(tn; dims=2)
  boundary_mps_top = contract_approx(
    tn, alg, BoundaryMPSDir"top_to_bottom"(); cutoff=cutoff, maxdim=maxdim
  )
  return BoundaryMPS(; bottom=reverse(boundary_mps_top.top))
end

function contract_approx(
  tn::Matrix{ITensor},
  alg::ContractionAlgorithm"boundary_mps",
  dirs::BoundaryMPSDir"left_to_right";
  cutoff,
  maxdim,
)
  tn = rotr90(tn)
  boundary_mps = contract_approx(
    tn, alg, BoundaryMPSDir"top_to_bottom"(); cutoff=cutoff, maxdim=maxdim
  )
  return BoundaryMPS(; left=boundary_mps.top)
end

function contract_approx(
  tn::Matrix{ITensor},
  alg::ContractionAlgorithm"boundary_mps",
  dirs::BoundaryMPSDir"right_to_left";
  cutoff,
  maxdim,
)
  tn = rotr90(tn)
  boundary_mps = contract_approx(
    tn, alg, BoundaryMPSDir"bottom_to_top"(); cutoff=cutoff, maxdim=maxdim
  )
  return BoundaryMPS(; right=boundary_mps.bottom)
end

function contract_approx(
  tn::Matrix{ITensor},
  alg::ContractionAlgorithm"boundary_mps",
  dirs::BoundaryMPSDir"all";
  cutoff,
  maxdim,
)
  mps_top = contract_approx(
    tn, alg, BoundaryMPSDir"top_to_bottom"(); cutoff=cutoff, maxdim=maxdim
  )
  mps_bottom = contract_approx(
    tn, alg, BoundaryMPSDir"bottom_to_top"(); cutoff=cutoff, maxdim=maxdim
  )
  mps_left = contract_approx(
    tn, alg, BoundaryMPSDir"left_to_right"(); cutoff=cutoff, maxdim=maxdim
  )
  mps_right = contract_approx(
    tn, alg, BoundaryMPSDir"right_to_left"(); cutoff=cutoff, maxdim=maxdim
  )
  return BoundaryMPS(;
    top=mps_top.top, bottom=mps_bottom.bottom, left=mps_left.left, right=mps_right.right
  )
end

function contract_approx(
  tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps"; dirs, cutoff, maxdim
)
  return contract_approx(tn, alg, BoundaryMPSDir(dirs); cutoff=cutoff, maxdim=maxdim)
end

# Compute the truncation projectors for the network,
# contracting from top to bottom
function contract_approx(
  tn::Matrix{ITensor}; alg="boundary_mps", dirs="all", cutoff=1e-8, maxdim=maxdim_arg(tn)
)
  return contract_approx(
    tn, ContractionAlgorithm(alg); dirs=dirs, cutoff=cutoff, maxdim=maxdim
  )
end

#
# Insert the projectors computed from the boundary MPS
#

default_projector_center(tn::Matrix) = (:, (size(tn, 2) + 1) ÷ 2)

# Split the links of the 2D tensor network in preperation for
# inserting MPS projectors.
function split_network(tn::Matrix{ITensor}; projector_center=default_projector_center(tn))
  @assert (projector_center[1] == :)
  tn_split = copy(tn)
  nrows, ncols = size(tn)
  @assert length(projector_center) == 2
  @assert projector_center[1] == Colon()
  projector_center_cols = projector_center[2]
  for ncol in 1:ncols
    if ncol ∉ projector_center_cols
      tn_split[:, ncol] = split_links(tn_split[:, ncol])
    end
  end
  return tn_split
end

# From an MPS, create a 1-site projector onto the MPS basis
function projector(x::MPS, projector_center)
  # Gauge the boundary MPS towards the projector_center column
  x = orthogonalize(x, projector_center)

  l = commoninds(get_itensor(x, projector_center - 1), x[projector_center])
  r = commoninds(get_itensor(x, projector_center + 1), x[projector_center])

  uₗ = x[1:(projector_center - 1)]
  uᵣ = reverse(x[(projector_center + 1):end])
  nₗ = length(uₗ)
  nᵣ = length(uᵣ)

  uₗᴴ = dag.(uₗ)
  uᵣᴴ = dag.(uᵣ)

  uₗ′ = reverse(prime.(uₗ))
  uᵣ′ = reverse(prime.(uᵣ))

  if !isempty(uₗ′)
    uₗ′[1] = replaceinds(uₗ′[1], l' => l)
  end
  if !isempty(uᵣ′)
    uᵣ′[1] = replaceinds(uᵣ′[1], r' => r)
  end

  Pₗ = vcat(uₗᴴ, uₗ′)
  Pᵣ = vcat(uᵣᴴ, uᵣ′)
  return Pₗ, Pᵣ
end

function insert_projectors(
  tn::Matrix{ITensor},
  boundary_mps::Vector{MPS},
  dirs::BoundaryMPSDir"top_to_bottom";
  projector_center,
)
  nrows, ncols = size(tn)
  @assert length(boundary_mps) == nrows - 1
  @assert all(x -> length(x) == ncols, boundary_mps)
  @assert length(projector_center) == 2
  @assert (projector_center[1] == :)
  projector_center_cols = projector_center[2]
  projectors = [projector(x, projector_center_cols) for x in boundary_mps]
  projectors_left = first.(projectors)
  projectors_right = last.(projectors)
  tn_split = split_network(tn; projector_center=projector_center)
  return tn_split, projectors_left, projectors_right
end

function insert_projectors(
  tn::Matrix{ITensor}, boundary_mps::Vector{MPS}; dirs, projector_center
)
  return insert_projectors(
    tn, boundary_mps, BoundaryMPSDir(dirs); projector_center=projector_center
  )
end

function insert_projectors(tn, boundary_mps::BoundaryMPS; center, projector_center=nothing)
  top_bottom = true
  if (center[1] == :)
    center = reverse(center)
    tn = rotr90(tn)
    top_bottom = false
  end

  @assert (center[2] == :)
  center_row = center[1]

  if isnothing(projector_center)
    projector_center = default_projector_center(tn)
  end
  @assert (projector_center[1] == :)

  # Approximately contract the tensor network.
  # Outputs a Vector of boundary MPS.
  #boundary_mps_top = contract_approx(tn; alg="boundary_mps", dirs="top_to_bottom", cutoff=cutoff, maxdim=maxdim)
  #boundary_mps_bottom = contract_approx(tn; alg="boundary_mps", dirs="bottom_to_top", cutoff=cutoff, maxdim=maxdim)

  boundary_mps_top = top_bottom ? boundary_mps.top : boundary_mps.left
  boundary_mps_bottom = top_bottom ? boundary_mps.bottom : boundary_mps.right

  # Insert approximate projectors into rows of the network
  boundary_mps_tot = vcat(
    boundary_mps_top[1:(center_row - 1)], dag.(boundary_mps_bottom[center_row:end])
  )
  tn_split, projectors_left, projectors_right = insert_projectors(
    tn, boundary_mps_tot; dirs="top_to_bottom", projector_center=projector_center
  )

  if !top_bottom
    tn_split = rotl90(tn_split)
  end
  return tn_split, projectors_left, projectors_right
end

function contraction_cache_top(tn, boundary_mps::Vector{MPS}, n)
  tn_cache = fill(ITensor(1.0), size(tn))
  for nrow in 1:size(tn, 1)
    if nrow == n
      tn_cache[nrow, :] .= boundary_mps[nrow]
    elseif nrow > n
      tn_cache[nrow, :] = tn[nrow, :]
    end
  end
  return tn_cache
end

function contraction_cache_top(tn, boundary_mps::Vector{MPS})
  tn_cache_top = Vector{typeof(tn)}(undef, length(boundary_mps))
  for n in 1:length(boundary_mps)
    tn_cache_top[n] = contraction_cache_top(tn, boundary_mps, n)
  end
  return tn_cache_top
end

function contraction_cache_bottom(tn, boundary_mps::Vector{MPS})
  tn_top = rot180(tn)
  tn_top = reverse(tn_top; dims=2)
  boundary_mps_top = reverse(boundary_mps)
  tn_cache_top = contraction_cache_top(tn_top, boundary_mps_top)
  tn_cache = reverse.(tn_cache_top; dims=2)
  tn_cache = rot180.(tn_cache)
  return tn_cache
end

function contraction_cache_left(tn, boundary_mps::Vector{MPS})
  tn_top = rotr90(tn)
  tn_cache_top = contraction_cache_top(tn_top, boundary_mps)
  tn_cache = rotl90.(tn_cache_top)
  return tn_cache
end

function contraction_cache_right(tn, boundary_mps::Vector{MPS})
  tn_bottom = rotr90(tn)
  tn_cache_bottom = contraction_cache_bottom(tn_bottom, boundary_mps)
  tn_cache = rotl90.(tn_cache_bottom)
  return tn_cache
end

function contraction_cache(tn, boundary_mps)
  cache_top = contraction_cache_top(tn, boundary_mps.top)
  cache_bottom = contraction_cache_bottom(tn, boundary_mps.bottom)
  cache_left = contraction_cache_left(tn, boundary_mps.left)
  cache_right = contraction_cache_right(tn, boundary_mps.right)
  return (top=cache_top, bottom=cache_bottom, left=cache_left, right=cache_right)
end

function boundary_mps_top(tn)
  return [MPS(tn[n][n, :]) for n in 1:length(tn)]
end

function boundary_mps_bottom(tn)
  tn_top = rot180.(tn)
  tn_top = reverse.(tn_top; dims=2)
  boundary_mps = boundary_mps_top(tn_top)
  return reverse(boundary_mps)
end

function boundary_mps_left(tn)
  tn_top = rotr90.(tn)
  return boundary_mps_top(tn_top)
end

function boundary_mps_right(tn)
  tn_bottom = rotr90.(tn)
  return boundary_mps_bottom(tn_bottom)
end

function boundary_mps(tn::NamedTuple)
  _boundary_mps_top = boundary_mps_top(tn.top)
  _boundary_mps_bottom = boundary_mps_bottom(tn.bottom)
  _boundary_mps_left = boundary_mps_left(tn.left)
  _boundary_mps_right = boundary_mps_right(tn.right)
  return BoundaryMPS(;
    top=_boundary_mps_top,
    bottom=_boundary_mps_bottom,
    left=_boundary_mps_left,
    right=_boundary_mps_right,
  )
end

# Return a network that when contracted equals the
# squared norm of the input network.
# TODO: rename norm2_network
function sqnorm(ψ::Matrix{ITensor})
  # Square the tensor network
  ψᴴ = addtags(linkinds, dag.(ψ), "bra")
  ψ′ = addtags(linkinds, ψ, "ket")
  return mapreduce(vec, vcat, (ψᴴ, ψ′))
end

# Return a network that approximates the squared norm of the
# input network by inserting projectors determined from approximately
# contracting the network with boundary MPS.
# TODO: rename norm2_network_approx
function sqnorm_approx(ψ::Matrix{ITensor}; center, cutoff, maxdim)
  # Square the tensor network
  ψᴴ = addtags(linkinds, dag.(ψ), "bra")
  ψ′ = addtags(linkinds, ψ, "ket")
  # TODO: implement contract(commoninds, ψ′, ψᴴ)
  tn = ψ′ .* ψᴴ

  # Contract in every direction
  combiner_gauge = combiners(linkinds, tn)
  tnᶜ = insert_gauge(tn, combiner_gauge)
  boundary_mpsᶜ = contract_approx(tnᶜ; maxdim=maxdim, cutoff=cutoff)

  tn_cacheᶜ = contraction_cache(tnᶜ, boundary_mpsᶜ)
  tn_cache = insert_gauge(tn_cacheᶜ, combiner_gauge)
  _boundary_mps = boundary_mps(tn_cache)

  #
  # Insert projectors horizontally (to measure e.g. properties
  # in a row of the network)
  #

  tn_projected = insert_projectors(tn, _boundary_mps; center=center)
  tn_split, Pl, Pr = tn_projected

  ψᴴ_split = split_network(ψᴴ)
  ψ′_split = split_network(ψ′)

  Pl_flat = reduce(vcat, Pl)
  Pr_flat = reduce(vcat, Pr)
  return mapreduce(vec, vcat, (ψᴴ_split, ψ′_split, Pl_flat, Pr_flat))
end
