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
    γ₁::Float64
    ϵ₁::Float64
    ρ₁::Float64
    γ₂::Float64
    ϵ₂::Float64
    ρ₂::Float64

    # Technology
    a₁::Float64
    a₂::Float64
    σ₁::Float64
    σ₂::Float64
    δ₁::Float64
    δ₂::Float64
    A₁::Float64
    A₂::Float64
    B₁::Float64
    B₂::Float64


    # "Skin in the game" constraint
    χ::Float64

    # Transfers
    τ::Float64
end

function BrunnermeierSannikov2016Model(; γ₁ = 2.0, γ₂ = 2., ϵ₁ = 1.5,
                                       ϵ₂ = 1.5, ρ₁ = 0.05, ρ₂ = .05,
                                       a₁ = 1, a₂ = .7, σ₁ = 0.03,
                                       σ₂ = 0.03, δ₁ = 0.1, δ₂ = 0.1,
                                       A₁ = 57., A₂, = 57.,
                                       B₁ = 5200., B₂ = 5200.,
                                       χ = 1., τ = .01)
    BrunnermeierSannikov2016Model(γ₁, γ₂, ϵ₁, ϵ₂,
                                  ρ₁, ρ₂, a₁, a₂,
                                  σ₁, σ₂, δ₁, δ₂, A, B, χ, τ)
end

function initialize_stategrid(m::BrunnermeierSannikov2016Model; η_n = 80)
  OrderedDict(:η => range(0.01, stop = 0.99, length = η_n))
end

function initialize_y(m::BrunnermeierSannikov2016Model, stategrid::OrderedDict)
  x = fill(1.0, length(stategrid[:η]))
  OrderedDict(:v₁ => x, :v₂ => x, :q => x)
end

function (m::BrunnermeierSannikov2016Model)(state::NamedTuple, y::NamedTuple)
    γ₁ = m.γ₁; γ₂ = m.γ₂; ϵ₁ = m.ϵ₁; ϵ₂ = m.ϵ₂;
    ρ₁ = m.ρ₁; ρ₂ = m.ρ₂; a₁ = m.a₁; a₂ = m.a₂;
    σ₁ = m.σ₁; σ₂ = m.σ₂; δ₁ = m.δ₁; δ₂ = m.δ₂;
    A₁ = m.A₁; A₂ = m.A₂; B₁ = m.B₁; B₂ = m.B₂;
    χ = m.χ; τ = m.τ;

    x = state.x
    v₁, v₁η, v₁ηη, v₂, v₂η, v₂ηη, q, qη = y.v₁, y.v₁η, y.v₁ηη, y.v₂, y.v₂η, y.v₂ηη, y.q, y.qη

    # Determine capital allocations
    ψ₁ = 1.
    ψ₂ = 0.

    # Internal investment function
    #     Φᵢ(ιᵢ) = (Aᵢ / Bᵢ) * log(ιᵢ / Bᵢ - 1) - δᵢ
    # with optimality condition
    #     Φ'ᵢ(ιᵢ) = 1 / q
    μ_K1 = (A₁ / B₁) * log(A₁ * q) - δ₁
    μ_K2 = (A₂ / B₂) * log(A₂ * q) - δ₂
    μ_K  = ψ₁ * μ_K1 + ψ₂ * μ_K2
    ι₁   = (A₁ * q - 1) / B₁
    ι₂   = (A₂ * q - 1) / B₂
    ι    = ψ₁ * ι₁ + ψ₂ * ι₂

    # Compute volatilities
    sigma_q = qp / q * ((psi1 - eta) * (psi1 * sigma1 + psi2 * sigma2) + sigma_q)
    σ_q = qη * (ψ₁ - η) * (ψ₁ * σ₁ + ψ₂ * σ₂) / (q - qη * (ψ₁ - η))
    σ_η = (ψ₁ / η - 1) * (ψ₁ * σ₁ + ψ₂ * σ₂ + σ_q)
    σ_v₁ = v₁η / v₁ * σ_η * η
    σ_v₂ = v₂η / v₂ * σ_η * η

  # Interest rate r
  μX = x * (1 - x) * ((σA * κ + νA * κν - 1 / pA - τ) - (σB * κ -  1 / pB + τ * x / (1 - x)) - (σA - σB) * (σ + σp))
  μpA = pAx / pA * μX + pAν / pA * μν + 0.5 * pAxx / pA * σX^2 + 0.5 * pAνν / pA * σν^2 + pAxν / pA * σX * σν
  μpB = pBx / pB * μX + pBν / pB * μν + 0.5 * pBxx / pB * σX^2 + 0.5 * pBνν / pB * σν^2 + pBxν / pB * σX * σν
  μp = px / p * μX + pν / p * μν + 0.5 * pxx / p * σX^2 + 0.5 * pνν / p * σν^2 + pxν / p * σX * σν
  r = (1 - i) / p + g + μp + σ * σp - κ * (σ + σp) - γ / x * (ϕ * ν)^2

    # Compute driftxi^(1-eps)
    v_A = xi * eta * q
    xi = v_A / eta / q
    μ_η = (a₁ - ι₁) / q - (v₁ / η / q)^(1 - ϵ₁) +
        (χ * ψ₁ / η - 1) * (γ₁ * ςr + (γ₁ - 1) * ς₁) +
        (1 - χ) * ψ₂ / η * (γ₁ * ς

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
