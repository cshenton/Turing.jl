#########################
# Sampler I/O Interface #
#########################

##########
# Sample #
##########

mutable struct Sample
    weight:: Float64 # particle weight
    value::Dict{Symbol,Any}
end

Base.getindex(s::Sample, v::Symbol) = getjuliatype(s, v)

getjuliatype(s::Sample, v::Symbol, cached_syms=nothing) = begin
  # NOTE: cached_syms is used to cache the filter entiries in svalue. This is helpful when the dimension of model is huge.
  if cached_syms == nothing
    # Get all keys associated with the given symbol
    syms = collect(Iterators.filter(k -> occursin(string(v)*"[", string(k)), keys(s.value)))
  else
    syms = collect((Iterators.filter(k -> occursin(string(v), string(k)), cached_syms)))
  end

  # Map to the corresponding indices part
  idx_str = map(sym -> replace(string(sym), string(v) => ""), syms)
  # Get the indexing component
  idx_comp = map(idx -> collect(Iterators.filter(str -> str != "", split(string(idx), [']','[']))), idx_str)

  # Deal with v is really a symbol, e.g. :x
  if isempty(idx_comp)
    @assert haskey(s.value, v)
    return Base.getindex(s.value, v)
  end

  # Construct container for the frist nesting layer
  dim = length(split(idx_comp[1][1], ','))
  if dim == 1
    sample = Vector(undef, length(unique(map(c -> c[1], idx_comp))))
  else
    d = max(map(c -> eval(parse(c[1])), idx_comp)...)
    sample = Array{Any, length(d)}(undef, d)
  end

  # Fill sample
  for i = 1:length(syms)
    # Get indexing
    idx = Main.eval(parse(idx_comp[i][1]))
    # Determine if nesting
    nested_dim = length(idx_comp[1]) # how many nested layers?
    if nested_dim == 1
      setindex!(sample, getindex(s.value, syms[i]), idx...)
    else  # nested case, iteratively evaluation
      v_indexed = Symbol("$v[$(idx_comp[i][1])]")
      setindex!(sample, getjuliatype(s, v_indexed, syms), idx...)
    end
  end
  sample
end

#########
# Chain #
#########

"""
    Turing specific MCMCChain.AbstractChains type.


Example:

```julia
# Define a model
@model gmodel(x) = begin
    mu ~ Normal(0, 1)
    sigma ~ InverseGamma(2, 3)
    for n in 1:length(x)
        x[n] ~ Normal(mu, sqrt(sigma))
    end
    return (mu, sigma)
end

# Run the inference engine
chain = sample(test([0., 0.2, -0.2]), SMC(1000))

# show the log model evidence
chain[:logevidence]

# show the weighted trajactory for `sigma`
chain[:sigma]

# find the mean of `mu`
mean(chain[:mu])
```
"""
struct Chain{T<:Real} <: AbstractChains
    logevidence::Vector{Float64}
    samples::Array{Sample}
    value::Array{Union{Missing, Float64}, 3}
    range::AbstractRange{Int}
    names::Vector{String}
    chains::Vector{Int}
    info::Dict{Symbol, Any}
end


"""
    Chain()

Construct an empty chain object.
"""
function Chain()
    c = Chain(
            zeros(Float64),
            Vector{Sample}(),
            Array{Union{Missing, Float64}, 3}(undef, 0, 0, 0),
            0:0,
            Vector{String}(),
            Vector{Int}(),
            Dict{Symbol,Any}()
    )
    return c
end

"""
    Chain(logevidence::Float64, value::Array{Sample})

Construct a chain object holding given samples.
"""
function Chain(logevidence::Float64, s::Array{Sample})
    return Chain([logevidence], s)
end

function Chain(logevidence::Vector{Float64}, s::Array{Sample})
     chn = Chain()
     chn.logevidence = logevidence
     chn.samples = deepcopy(s)
     flatten!(chn)

     return chn
end

function flatten!(chn::Chain)
    ## Flatten samples into Mamba's chain type.
    Nsamples = length(chn.samples)
    names_ = Vector{Array{AbstractString}}(undef, Nsamples)
    vals_  = Vector{Array}(undef, Nsamples)

    for (i, s) in enumerate(chn.samples)
        v, n = flatten(s)
        vals_[i] = v
        names_[i] = n
    end

    # Assuming that names[i] == names[j] for all (i,j)
    P = length(names[1])
    v_ = Array{Union{Missing, Float64}, 3}(undef, Nsamples, P, 1)
    for n in 1:Nsamples
        for p in 1:P
            v_[n, p, 1] = vals_[n][p]
        end
    end

    chn.value = c.value
    chn.range = range(1, step = 1, length = Nsamples)
    chn.names = names_[1]
    chn.chains = collect(1:1)

    return chn
end

# ind2sub is deprecated in Julia 1.0
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])

function flatten(s::Sample)
    vals  = Vector{Float64}()
    names = Vector{AbstractString}()
    for (k, v) in s.value
        flatten!(names, vals, string(k), v)
    end
    return vals, names
end

function flatten!(names, value::Array{Float64}, k::String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, Array)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = k * string(ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                isa(v[i], Nothing) && println(v, i, v[i])
                push!(value, Float64(v[i]))
                push!(names, name)
            elseif isa(v[i], Array)
                name = k * string(ind2sub(size(v), i))
                flatten(names, value, name, v[i])
            else
                error("Unknown var type: typeof($v[i])=$(typeof(v[i]))")
            end
        end
    else
        error("Unknown var type: typeof($v)=$(typeof(v))")
    end
end

function Base.getindex(c::Chain, v::Symbol)
    # This strange implementation is mostly to keep backward compatability.
    #  Needs some refactoring a better format for storing results is available.
    if v == :logevidence
        return c.:logevidence
    elseif v==:samples
        return c.samples
    elseif v==:logweights
        return c[:lp]
    else
        return map((s)->Base.getindex(s, v), c.samples)
    end
end

function Base.getindex(c::Chain, expr::Expr)
    str = replace(string(expr), r"\(|\)" => "")
    @assert match(r"^\w+(\[(\d\,?)*\])+$", str) != nothing "[Turing.jl] $expr invalid for getindex(::Chain, ::Expr)"
    return c[Symbol(str)]
end

function Base.vcat(c1::Chain, args::Chain...)

    all(c -> c.names == c1.names, args) || throw(ArgumentError("chain names differ"))
    all(c -> c.chains == c1.chains, args) || throw(ArgumentError("sets of chains differ"))

    le_ = cat(1, c1.logevidence, map(c -> c.logevidence, args)...)
    s_ = cat(1, c1.samples, map(c -> c.samples, args)...)

    return Chain(le_, s_)
end

"""
    save!(c::Chain, spl::Sampler, model::Function, vi)

Store sampler, model function and sampling state (vi) into a chain object.
"""
function save!(c::Chain, spl::Sampler, model::Function, vi)
    c.info[:spl] = spl
    c.info[:model] = model
    c.info[:vi] = deepcopy(vi)
end

"""
    resume(c::Chain, n_iter::Int)

Wrapper function to resume a MCMC sampling.
"""
function resume(c::Chain, n_iter::Int)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"
    return sample(c.info[:model], c.info[:spl].alg; resume_from=c, reuse_spl_n=n_iter)
end
