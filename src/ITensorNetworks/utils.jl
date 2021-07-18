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
