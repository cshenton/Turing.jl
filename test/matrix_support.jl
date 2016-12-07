using Distributions
using Turing

@model hmcmatrixsup begin
  @assume p ~ Wishart(7, [1 0.5; 0.5 1])
  @predict p
end

sample(hmcmatrixsup, HMC(10, 0.1, 3))