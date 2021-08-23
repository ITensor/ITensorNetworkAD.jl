using ITensors, ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: PEPS, Models
using ITensorNetworkAD.Optimizations: gd_error_tracker

function test_peps_gd()
  Ny, Nx = 3, 4
  num_sweeps = 20
  cutoff = 1e-15
  maxdim = 1000
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  return gd_error_tracker(
    peps, H_line; stepsize=0.005, num_sweeps=num_sweeps, cutoff=cutoff, maxdim=maxdim
  )
end

test_peps_gd()
