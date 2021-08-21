#
# Boundary MPS contraction methods
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

function boundary_mps(tn::Matrix{ITensor}; cutoff, maxdim)
  #TODO
  # Contract in every direction
  combiner_gauge = combiners(linkinds, tn)
  tnᶜ = insert_gauge(tn, combiner_gauge)
  boundary_mpsᶜ = contract_approx(tnᶜ; maxdim=maxdim, cutoff=cutoff)
  tn_cacheᶜ = contraction_cache(tnᶜ, boundary_mpsᶜ)
  tn_cache = insert_gauge(tn_cacheᶜ, combiner_gauge)
  return boundary_mps(tn_cache)
end
