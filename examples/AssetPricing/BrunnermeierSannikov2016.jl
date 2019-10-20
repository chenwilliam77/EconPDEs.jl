using EconPDEs, Distributions

# This script implements a generalization of the model presented
# in the Handbook of Macro Vol. 2 chapter
# "Macro, Money, and Finance: A Continuous Time Approach"
# by Brunnermeier and Sannikov.

# The model includes agent-specific risk aversion, EIS,
# discount rates, depreciation, consumption good productivity,
# and volatility of capital holdings.

# macro instantiate_parameters(m)
#     for name in fieldnames(m)
#         eval(:$(name) = m.$(name))
#     end
# end

mutable struct BrunnermeierSannikov2016Model
    # Utility Function
    γ₁::Float64
    γ₂::Float64
    ϵ₁::Float64
    ϵ₂::Float64
    ρ₁::Float64
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
    χ_::Float64

    # Transfers
    τ::Float64
end

function BrunnermeierSannikov2016Model(; γ₁ = 5., γ₂ = 5., ϵ₁ = 2.,
                                       ϵ₂ = 2., ρ₁ = .05, ρ₂ = .05,
                                       a₁ = 1, a₂ = .7, σ₁ = 0.03,
                                       σ₂ = 0.03, δ₁ = 0.05, δ₂ = 0.05,
                                       A₁ = 53.2, A₂ = 53.2,
                                       B₁ = 0., B₂ = 0.,
                                       χ_ = 1., τ = .01)
    BrunnermeierSannikov2016Model(γ₁, γ₂, ϵ₁, ϵ₂,
                                  ρ₁, ρ₂, a₁, a₂,
                                  σ₁, σ₂, δ₁, δ₂, A₁, A₂, B₁, B₂, χ_, τ)
end

function initialize_stategrid(m::BrunnermeierSannikov2016Model; η_n = 80)
  OrderedDict(:η => range(0.01, stop = 0.99, length = η_n))
end

function initialize_y(m::BrunnermeierSannikov2016Model, stategrid::OrderedDict)
  x = fill(1.0, length(stategrid[:η]))
  OrderedDict(:ξ₁ => x,
              :ξ₂ => x)
end

function (m::BrunnermeierSannikov2016Model)(state::NamedTuple, y::NamedTuple)
    γ₁ = m.γ₁; γ₂ = m.γ₂; ϵ₁ = m.ϵ₁; ϵ₂ = m.ϵ₂;
    ρ₁ = m.ρ₁; ρ₂ = m.ρ₂; a₁ = m.a₁; a₂ = m.a₂;
    σ₁ = m.σ₁; σ₂ = m.σ₂; δ₁ = m.δ₁; δ₂ = m.δ₂;
    A₁ = m.A₁; A₂ = m.A₂; B₁ = m.B₁; B₂ = m.B₂;
    χ_ = m.χ_; τ = m.τ;

    η = state.η
    # ξ₁, ξ₁η, ξ₁ηη, ξ₂, ξ₂η, ξ₂ηη, q, qη, qηη = y.ξ₁, y.ξ₁η, y.ξ₁ηη, y.ξ₂, y.ξ₂η, y.ξ₂ηη, y.q, y.qη, y.qηη
    ξ₁, ξ₁η, ξ₁ηη, ξ₂, ξ₂η, ξ₂ηη = y.ξ₁, y.ξ₁η, y.ξ₁ηη, y.ξ₂, y.ξ₂η, y.ξ₂ηη

    # Determine capital allocations
    ψ₁ = 1.
    ψ₂ = 0.
    q = (a₁ * ψ₁ + a₂ * ψ₂) / (ρ₁ * η + ρ₂ * (1 - η))

    # Internal investment function
    #     Φᵢ(ιᵢ) = (Aᵢ / Bᵢ) * log(ιᵢ / Bᵢ - 1) - δᵢ
    # with optimality condition
    #     Φ'ᵢ(ιᵢ) = 1 / q
    # μ_K1 = (A₁ / B₁) * log(A₁ * q) - δ₁
    # μ_K2 = (A₂ / B₂) * log(A₂ * q) - δ₂
    # μ_K1 = (q - B₁) / (2 * A₁) - δ₁
    # μ_K2 = (q - B₂) / (2 * A₂) - δ₂
    # μ_K  = ψ₁ * μ_K1 + ψ₂ * μ_K2
    μ_K = 0.02
    μ_K1 = 0.02
    μ_K2 = 0.02
    # ι₁   = (A₁ * q - 1) / B₁
    # ι₂   = (A₂ * q - 1) / B₂
    # ι₁ = A₁ * ((q - B₁) / (2 * A₁))^2 + B₁ * ((q - B₁) / (2 * A₁))
    # ι₂ = A₂ * ((q - B₂) / (2 * A₂))^2 + B₂ * ((q - B₂) / (2 * A₂))
    # ι    = ψ₁ * ι₁ + ψ₂ * ι₂
    ι = 0.
    ι₁ = 0.

    # Compute volatilities
    # σ_q  = qη * (ψ₁ - η) * (ψ₁ * σ₁ + ψ₂ * σ₂) / (q - qη * (ψ₁ - η))
    σ_q = 0.
    σ_η  = ψ₁ / η * (σ₁ + σ_q) - σ_q - (ψ₁ * σ₁ + ψ₂ * σ₂)
    σ_ξ₁ = ξ₁η / ξ₁ * σ_η * η
    σ_ξ₂ = ξ₂η / ξ₂ * σ_η * η
    # σ_ξ₁ = σ_v₁ - σ_η - σ_q
    # σ_ξ₂ = σ_v₁ + η / (1 - η) * σ_η - σ_q

    # Compute risk premia
    ς_K₁  = ψ₁ / η * (σ₁ + σ_q)^2
    ς_ξ₁ = (σ₁ + σ_q) * σ_ξ₁

    # Consumption market-clearing
    c₁_per_n₁ = (ϵ₁ == 1.) ? ρ₁ : ξ₁^(1 - ϵ₁)
    c₂_per_n₂ = (ϵ₂ == 1.) ? ρ₂ : ξ₂^(1 - ϵ₂)
    # ∂q_∂t     = q * (c₁_per_n₁ * η + c₂_per_n₂ * (1 - η)) - (a₁ * ψ₁ + a₂ * ψ₂ - ι)
    ∂q_∂t = 0.

    # Compute drift
    μ_η = (a₁ - ι₁) / q - c₁_per_n₁ - τ +
        (ψ₁ / η - 1) * (γ₁ * ς_K₁ + (γ₁ - 1) * ς_ξ₁) -
        (ψ₁ / η * (σ₁ + σ_q) - (ψ₁ * σ₁ + ψ₂ * σ₂ + σ_q)) * (ψ₁ * σ₁ + ψ₂ * σ₂ + σ_q)

    # Compute interest rate
    # μ_q = qη / q * μ_η * η + (1 / 2) * qηη / q * (η * σ_η)^2
    μ_q = 0.
    E_returns = (a₁ - ι₁) / q + μ_q + μ_K1 + σ₁ * σ_q
    r_f = E_returns - γ₁ * ς_K₁ - (γ₁ - 1) * ς_ξ₁

    # Evolution of marginal value of agents' net worth
    μ_ξ₁ = ξ₁η / ξ₁ * μ_η * η + (1 / 2) * ξ₁ηη / ξ₁ * (η * σ_η)^2 # Drift from Ito's lemma
    μ_ξ₂ = ξ₂η / ξ₂ * μ_η * η + (1 / 2) * ξ₂ηη / ξ₂ * (η * σ_η)^2
    c₁_term = (ϵ₁ == 1.) ? 1 / (1 - 1 / ϵ₁) * (ξ₁^(1 - ϵ₁) - ρ₁) : ρ₁ * log(ρ₁ / ξ₁)
    c₂_term = (ϵ₂ == 1.) ? 1 / (1 - 1 / ϵ₂) * (ξ₂^(1 - ϵ₂) - ρ₂) : ρ₂ * log(ρ₂ / ξ₂)

    ∂ξ₁_∂t = ξ₁ * (c₁_term + μ_ξ₁ + r_f -
        c₁_per_n₁ - τ + γ₁ / 2 * ((ψ₁ / η * (σ₁ + σ_q))^2 - σ_ξ₁^2))
    ∂ξ₂_∂t = 0. #ξ₂ * (c₂_term + μ_ξ₂ + r_f -
        # c₂_per_n₂ + τ * η / (1 - η) - γ₂ / 2 * σ_ξ₂^2)

 return (∂ξ₁_∂t, ∂ξ₂_∂t), (μ_η), (μ_η = μ_η, q = q, ξ₁ = ξ₁, ξ₂ = ξ₂,
                                         ψ₁ = ψ₁, ψ₂ = ψ₂,
                                         μ_K = μ_K, ι = ι, σ_q = σ_q,
                                         σ_η = σ_η, σ_ξ₁ = σ_ξ₁, σ_ξ₂ = σ_ξ₂,
                                         ς_K₁ = ς_K₁, ς_ξ₁ = ς_ξ₁)
end

m = BrunnermeierSannikov2016Model(ϵ₁ = 1., ϵ₂ = 1., γ₁ = 1., γ₂ = 1.)
stategrid = initialize_stategrid(m)
y0 = initialize_y(m, stategrid)
y, result, distance = pdesolve(m, stategrid, y0)
# y, result, distance = pdesolve(m, stategrid, y0; is_algebraic = OrderedDict(:v₁ => false, :v₂ => false, :q => true))
