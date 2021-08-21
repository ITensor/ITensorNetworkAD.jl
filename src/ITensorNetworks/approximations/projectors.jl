"""Insert the projectors computed from the boundary MPS"""

default_projector_center(tn::Matrix) = (:, (size(tn, 2) + 1) ÷ 2)

# Split the links of the 2D tensor network in preperation for
# inserting MPS projectors.
function split_network(
  tn::Matrix{ITensor}; projector_center=default_projector_center(tn), rotation=false
)
  tn = rotation ? rotr90(tn) : tn
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
  return rotation ? rotr90(tn_split, -1) : tn_split
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
  # Insert projectors horizontally (to measure e.g. properties
  # in a row of the network)
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  tn_projected = insert_projectors(tn, bmps; center=center)
  tn_split, Pl, Pr = tn_projected
  ψᴴ_split = split_network(ψᴴ)
  ψ′_split = split_network(ψ′)
  Pl_flat = reduce(vcat, Pl)
  Pr_flat = reduce(vcat, Pr)
  return mapreduce(vec, vcat, (ψᴴ_split, ψ′_split, Pl_flat, Pr_flat))
end
