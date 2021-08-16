using ITensors, ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: PEPS, Models
using ITensorNetworkAD.Optimizations: gd_error_tracker

@testset "test error of approximate peps" begin
  Nx, Ny = 3, 3
  num_sweeps = 20
  cutoff = 1e-15
  maxdim = 100
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  gd_error_tracker(
    peps, H_line; stepsize=0.005, num_sweeps=num_sweeps, cutoff=cutoff, maxdim=maxdim
  )
end
