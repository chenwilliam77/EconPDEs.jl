using EconPDEs, Distributions, Parameters

# This script implements a generalization of the model presented
# in the Handbook of Macro Vol. 2 chapter
# "Macro, Money, and Finance: A Continuous Time Approach"
# by Brunnermeier and Sannikov.

# The model includes agent-specific risk aversion, EIS,
# discount rates, depreciation, consumption good productivity,
# and volatility of capital holdings.

macro instantiate_parameters(m)
    for name in fieldnames(m)
        eval(:$(name) = $(m.name))
    end
end

mutable struct BrunnermeierSannikov2016Model
  # Utility Function
  γₑ::Float64
  ψₑ::Float64
  ρₑ::Float64
  γₕ::Float64
  ψₕ::Float64
  ρₕ::Float64

  # Technology
  aₑ::Float64
  aₕ::Float64
  σₑ::Float64
  σₕ::Float64
  δₑ::Float64
  δₕ::Float64
  A::Float64
  B::Float64

  # "Skin in the game" constraint
  χ::Float64
end

function BrunnermeierSannikov2016Model(; γₑ = 2.0, γₕ = 2., ψₑ = 1.5, ψₕ = 1.5, ρₑ = 0.05, ρₕ = .05,
                                       aₑ = 1, aₕ = .7, σₑ = 0.03, σₕ = 0.03, δₑ = 0.1, δₕ = 0.1, A = 57., B = 5200.,
                                       χ = 1.)
    BrunnermeierSannikov2016Model(γₑ = 2.0, γₕ = 2., ψₑ = 1.5, ψₕ = 1.5, ρₑ = 0.05, ρₕ = .05,
                                  aₑ = 1, aₕ = .7, σₑ = 0.03, σₕ = 0.03, δₑ = 0.1, δₕ = 0.1, A = 57., B = 5200.,
                                  χ = 1.)
end

function initialize_stategrid(m::DiTellaModel; η_n = 80)
  OrderedDict(:η => range(0.001, stop = 0.999, length = η_n))
end

function initialize_y(m::BrunnermeierSannikov2016Model, stategrid::OrderedDict)
  x = fill(1.0, length(stategrid[:η]))
  OrderedDict(:vₑ => x, :vₕ => x, :q => x)
end

function (m::BrunnermeierSannikov2016Model)(state::NamedTuple, y::NamedTuple)
    γₑ = m.γₑ; γₕ = m.γₕ; ψₑ = m.


  x, ν = state.x, state.ν
  pA, pAx, pAν, pAxx, pAxν, pAνν, pB, pBx, pBν, pBxx, pBxν, pBνν, p, px, pν, pxx, pxν, pνν = y.pA, y.pAx, y.pAν, y.pAxx, y.pAxν, y.pAνν, y.pB, y.pBx, y.pBν, y.pBxx, y.pBxν, y.pBνν, y.p, y.px, y.pν, y.pxx, y.pxν, y.pνν

  # drift and volatility of state variable ν
  g = p / (2 * A)
  i = A * g^2
  μν = κν * (νbar - ν)
  σν = σνbar * sqrt(ν)

  # Market price of risk κ
  σX = x * (1 - x) * (1 - γ) / (γ * (ψ - 1)) * (pAν / pA - pBν / pB) * σν / (1 - x * (1 - x) * (1 - γ) / (γ * (ψ - 1)) * (pAx / pA - pBx / pB))
  σpA = pAx / pA * σX + pAν / pA * σν
  σpB = pBx / pB * σX + pBν / pB * σν
  σp = px / p * σX + pν / p * σν
  κ = (σp + σ - (1 - γ) / (γ * (ψ - 1)) * (x * σpA + (1 - x) * σpB)) / (1 / γ)
  κν = γ * ϕ * ν / x
  σA = κ / γ + (1 - γ) / (γ * (ψ - 1)) * σpA
  νA = κν / γ
  σB = κ / γ + (1 - γ) / (γ * (ψ - 1)) * σpB

  # Interest rate r
  μX = x * (1 - x) * ((σA * κ + νA * κν - 1 / pA - τ) - (σB * κ -  1 / pB + τ * x / (1 - x)) - (σA - σB) * (σ + σp))
  μpA = pAx / pA * μX + pAν / pA * μν + 0.5 * pAxx / pA * σX^2 + 0.5 * pAνν / pA * σν^2 + pAxν / pA * σX * σν
  μpB = pBx / pB * μX + pBν / pB * μν + 0.5 * pBxx / pB * σX^2 + 0.5 * pBνν / pB * σν^2 + pBxν / pB * σX * σν
  μp = px / p * μX + pν / p * μν + 0.5 * pxx / p * σX^2 + 0.5 * pνν / p * σν^2 + pxν / p * σX * σν
  r = (1 - i) / p + g + μp + σ * σp - κ * (σ + σp) - γ / x * (ϕ * ν)^2

  # Market Pricing
  pAt = pA * (1 / pA  + (ψ - 1) * τ / (1 - γ) * ((pA / pB)^((1 - γ) / (1 - ψ)) - 1) - ψ * ρ + (ψ - 1) * (r + κ * σA + κν * νA) + μpA - (ψ - 1) * γ / 2 * (σA^2 + νA^2) + (2 - ψ - γ) / (2 * (ψ - 1)) * σpA^2 + (1 - γ) * σpA * σA)
  pBt = pB * (1 / pB - ψ * ρ + (ψ - 1) * (r + κ * σB) + μpB - (ψ - 1) * γ / 2 * σB^2 + (2 - ψ - γ) / (2 * (ψ - 1)) * σpB^2 + (1 - γ) * σpB * σB)
  # algebraic constraint
  pt = p * ((1 - i) / p - x / pA - (1 - x) / pB)

  return (pAt, pBt, pt), (μX, μν), (μX = μX, μν = μν, p = p, pA = pA, pB = pB, κ = κ, r = r, σX = σX)
end

m = DiTellaModel()
stategrid = initialize_stategrid(m)
y0 = initialize_y(m, stategrid)
y, result, distance = pdesolve(m, stategrid, y0)
y, result, distance = pdesolve(m, stategrid, y0; is_algebraic = OrderedDict(:pA => false, :pB => false, :p => true))
