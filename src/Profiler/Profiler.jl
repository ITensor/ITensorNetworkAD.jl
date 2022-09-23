module Profiler

export @profile, do_profile, profile_exit

_TIMEIT = false
TIME_DICT = Dict()
NUM_CALLS_DICT = Dict()
DETAIL_INFO = false

function do_profile(timeit=false)
  return global _TIMEIT = timeit
end

macro profile(func)
  name = func.args[1].args[1]
  hiddenname = gensym()
  func.args[1].args[1] = hiddenname
  @info "decorating $name" hiddenname

  _decorator(f) = (args...; kwargs...) -> begin
    function f_timer(args...; kwargs...)
      tstart = time()
      out = f(args...; kwargs...)
      tend = time()
      dt = tend - tstart
      if DETAIL_INFO
        @info "$(name) took [$(dt)] seconds"
      end
      if haskey(NUM_CALLS_DICT, name)
        NUM_CALLS_DICT[name] += 1
        TIME_DICT[name] += dt
      else
        NUM_CALLS_DICT[name] = 1
        TIME_DICT[name] = dt
      end
      return out
    end

    if _TIMEIT
      return f_timer(args...; kwargs...)
    else
      return f(args...; kwargs...)
    end
  end

  return esc(
    quote
      $func
      $name = $_decorator($hiddenname)
    end,
  )
end

function profile_exit()
  if length(TIME_DICT) > 0
    @info "---profiling info---"
    for (funcname, time) in reverse(sort(collect(TIME_DICT); by=x -> x[2]))
      @info "Calling $(funcname) $(NUM_CALLS_DICT[funcname]) times, overall time [$(time)]."
    end
  end
  global _TIMEIT = false
  global TIME_DICT = Dict()
  return global NUM_CALLS_DICT = Dict()
end

end  # module
