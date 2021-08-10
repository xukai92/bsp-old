using Setfield

# E-Step

Base.@kwdef struct ImportanceSampling
    nsamples::Int
    rsamples::Int = -1
end

function estep(alg::ImportanceSampling, ScenarioModel, scenarios, getforce::Function, like; verbose=false)
    latents = map(scenarios) do scenario
        obsworld = ScenarioModel(scenario, missing, getforce, like)
        chain = sample(obsworld, IS(), alg.nsamples; progress=verbose)
        samples = rsample(chain, alg.rsamples)
    end
    return latents
end

estep(alg::ImportanceSampling, ScenarioModel, scenarios, fm, like; kwargs...) = 
    estep(alg, ScenarioModel, scenarios, make_getforce(fm), like; kwargs...)

# M-Step

using Optim

abstract type AbstractForceModel end

struct ZeroForce <: AbstractForceModel end

function make_getforce(::ZeroForce)
    getforce(::Entity, state) = zeros(size(state.position)...)
    return getforce
end

struct OracleForce <: AbstractForceModel end

make_getforce(alg::OracleForce) = BayesianSymbolic.getforce

Base.@kwdef struct HandCodedForce{C, O} <: AbstractForceModel
    # NOTE It seems that zeros is more stable than ones
    constant::C = zeros(5)
    mask::Vector{Bool} = ones(Bool, 5)
    opt::O = LBFGS()
    niterations::Int
end

make_constant_kwargs(constant) =
    (C=constant[1], G₀=constant[2], G=constant[3], Gm=constant[4], Gc=constant[5])

function make_getforce(alg::HandCodedForce; constant=alg.constant)
    return (e, s) -> BayesianSymbolic.getforce_pairwise(e, s; make_constant_kwargs(constant .* alg.mask)...)
end

function mstep(alg::HandCodedForce, ScenarioModel, scenarios, latents, like; verbose=false)
    function _f(x)
        getforce = make_getforce(alg; constant=x)
        return -compute_logjoint(ScenarioModel, scenarios, latents, getforce, like)
    end
    # NOTE @btime => 4.342 s
    #retval = Optim.optimize(_f, alg.constant, alg.opt, Optim.Options(iterations=alg.niterations))
    # NOTE @btime => 373.0 ms
    od = OnceDifferentiable(_f, alg.constant; autodiff=:forward)
    retval = Optim.optimize(od, alg.constant, alg.opt, Optim.Options(iterations=alg.niterations))

    constant = Optim.minimizer(retval)
    verbose && @info "Hand-coded force constants" make_constant_kwargs(constant .* alg.mask)...
    return @set alg.constant = constant
end

Base.@kwdef struct BSPForce{C, T, G, O, L, E, B} <: AbstractForceModel
    grammar::G
    # NOTE It seems that zeros is more stable than ones
    constant::C = zeros(3)
    tree::T = nothing
    opt::O
    loweropt::L = (constant=zeros(3), opt=LBFGS(), niterations=3)
    external::E = nothing
    beta::B = 0
end

const EXTERNAL_GRAVITY = (e, s) -> [0, -9.8] .* e.mass'

function make_getforce(alg::BSPForce; constant=alg.constant, tree=alg.tree)
    getforce_with_constant = gen_getforce_with_constant(alg.grammar, tree; external=alg.external)
    return (e, s) -> getforce_with_constant(e, s, constant)
end

function gen_lossfunc_with_constant(external, beta, grammar::Grammar, tree::RuleNode, ScenarioModel, like)
    getforce_with_constant = gen_getforce_with_constant(grammar, tree; external=external)
    function lossfunc_with_constant(grammar::Grammar, tree::RuleNode, constant, scenarios, latents)
        pcfg = ProbabilisticGrammar(grammar)
        loss_complexity = -logpdf(pcfg, tree)
        getforce = (e, s) -> getforce_with_constant(e, s, constant)
        loss_fitness = -compute_logjoint(ScenarioModel, scenarios, latents, getforce, like)
        loss_total = beta * loss_complexity + loss_fitness
        return loss_total
    end
    return lossfunc_with_constant
end

function find_constant(
    cfg, lossfunc_with_constant, grammar::Grammar, tree::RuleNode, scenarios, latents
)
    _f(x) = lossfunc_with_constant(grammar, tree, x, scenarios, latents)
    od = OnceDifferentiable(_f, cfg.constant; autodiff=:forward)

    retval = Optim.optimize(od, cfg.constant, cfg.opt, Optim.Options(iterations=cfg.niterations))
    return Optim.minimizer(retval)
end

function gen_lossfunc(loweropt, external, beta, ScenarioModel, scenarios, latents, like)
    function lossfunc(tree::RuleNode, grammar::Grammar; return_constant::Bool=false)
        lossfunc_with_constant = gen_lossfunc_with_constant(external, beta, grammar, tree, ScenarioModel, like)
        constant = find_constant(loweropt, lossfunc_with_constant, grammar, tree, scenarios, latents)
        if return_constant
            return constant
        end
        return lossfunc_with_constant(grammar, tree, constant, scenarios, latents)
    end
    return lossfunc
end

function mstep(alg::BSPForce, ScenarioModel, scenarios, latents, like; verbose=false)
    lossfunc = gen_lossfunc(alg.loweropt, alg.external, alg.beta, ScenarioModel, scenarios, latents, like)
    retval = ExprOptimization.optimize(alg.opt, alg.grammar, :Force, lossfunc; verbose=verbose)

    constant = lossfunc(retval.tree, alg.grammar; return_constant=true)
    verbose && @info "BSP force constants" constant
    return @set(@set(alg.constant = constant).tree = retval.tree)
end
