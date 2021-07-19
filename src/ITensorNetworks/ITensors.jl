#
# ITensors.jl extensions
#

# Generalize siteind to n-dimensional lattice
function ITensors.siteind(st::SiteType, N1::Integer, N2::Integer, Ns::Integer...; kwargs...)
  s = siteind(st; kwargs...)
  if !isnothing(s)
    ts = "n1=$N1,n2=$N2"
    for i in eachindex(Ns)
      ts *= ",n$(i + 2)=$(Ns[i])"
    end
    return addtags(s, ts)
  end
  return isnothing(s) && error(space_error_message(st))
end

# Generalize siteinds to n-dimensional lattice
function ITensors.siteinds(
  str::AbstractString, N1::Integer, N2::Integer, Ns::Integer...; kwargs...
)
  st = SiteType(str)
  return [siteind(st, ns...) for ns in Base.product(1:N1, 1:N2, UnitRange.(1, Ns)...)]
end

using ITensors: data

# Used for sorting a collection of Index
function Base.isless(i1::Index, i2::Index)
  return isless(
    (id(i1), plev(i1), tags(i1), dir(i1)), (id(i2), plev(i2), tags(i2), dir(i2))
  )
end

# Used for sorting a collection of Index
Base.isless(ts1::TagSet, ts2::TagSet) = isless(data(ts1), data(ts2))

# Get the promoted type of the Index objects in a collection
# of Index (Tuple, Vector, ITensor, etc.)
indtype(i::Index) = typeof(i)
indtype(T::Type{<:Index}) = T
indtype(is::Tuple{Vararg{<:Index}}) = eltype(is)
indtype(is::Vector{<:Index}) = eltype(is)
indtype(A::ITensor...) = indtype(inds.(A))

indtype(tn1, tn2) = promote_type(indtype(tn1), indtype(tn2))
indtype(tn) = mapreduce(indtype, promote_type, tn)

#
# MPS functionality extensions
#

Base.keytype(m::MPS) = keytype(data(m))

# A version of indexing which returns an empty order-0 ITensor
# when out of bounds
get_itensor(x::MPS, n::Int) = n in 1:length(x) ? x[n] : ITensor()

# Reverse the site ordering of an MPS.
# XXX: also reverse the orthogonality limits.
Base.reverse(x::MPS) = MPS(reverse(x.data))

