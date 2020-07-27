using Flux
using Flux: throttle
using Flux.Losses: ctc
using BSON
using ProgressBars
using LinearAlgebra
using Statistics
using DelimitedFiles

const TRAINDIR = "train"
const TESTDIR = "test"

const EPOCHS = 150

m = Chain(LSTM(26, 186),
          Dense(186, 62))
# m = gpu(m)

function loss(x, y)
  yhat = m.(x)
  yhat = reduce(hcat, yhat)
  l = ctc(gpu(yhat), y)
  Flux.reset!(m)
  return l
end

function readData(dataDir)
  fnames = readdir(dataDir)[1:101]
  Xs = []
  Ys = []

  for fname in ProgressBar(fnames)
    BSON.@load joinpath(dataDir, fname) x y
    x = [x[i,:] for i in 1:size(x,1)]
    push!(Xs, x)
    push!(Ys, Array(y'))
  end

  return (Xs, Ys)
end

function lev(s, t)
    m = length(s)
    n = length(t)
    d = Array{Int}(zeros(m+1, n+1))

    for i=2:(m+1)
        @inbounds d[i, 1] = i-1
    end

    for j=2:(n+1)
        @inbounds d[1, j] = j-1
    end

    for j=2:(n+1)
        for i=2:(m+1)
            @inbounds if s[i-1] == t[j-1]
                substitutionCost = 0
            else
                substitutionCost = 1
            end
            @inbounds d[i, j] = min(d[i-1, j] + 1, # Deletion
                            d[i, j-1] + 1, # Insertion
                            d[i-1, j-1] + substitutionCost) # Substitution
        end
    end

    @inbounds return d[m+1, n+1]
end

function collapse(seq)
  s = [x for x in seq if x != 62]
  if isempty(s) return s end
  s = [seq[1]]
  for ch in seq[2:end]
    if ch != s[end] && ch != 62
      push!(s, ch)
    end
  end
  return s
end

function per(x, y)
  yhat = m.(x)
  yhat = reduce(hcat, yhat)
  Flux.reset!(m)
  yhat = mapslices(argmax, yhat, dims=1) |> vec |> collapse
  y = mapslices(argmax, y, dims=1) |> vec |> collapse
  return lev(yhat, y) / length(y)
end

function main()
  println("Loading files")
  Xs, Ys = readData(TRAINDIR)
  # Xs = gpu.(Xs)
  # Ys = gpu.(Ys)
  data = collect(zip(Xs, Ys))

  # valData = data[1:184]
  # data = data[185:end]
  # loss(data[1]...)

  evalcb = () -> @show loss(data[100]...)
  
  println("Beginning training")

  opt = ADAM()
  for i in 1:EPOCHS
    println("Beginning epoch $i/$EPOCHS")
    Flux.train!(loss, params(m), data, opt, cb = throttle(evalcb, 10))
    # Flux.train!(loss, params(m), data, opt)
    p = mean(map(x -> per(x...), data))
    println("PER: $(p*100)")
  end

end

main()
