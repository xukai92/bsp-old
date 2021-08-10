using InteractiveUtils, DrWatson, Comonicon
if isdefined(Main, :IJulia) && Main.IJulia.inited
    using Revise
else
    ENV["GKSwstype"] = 100 # suppress warnings during gif saving
end
versioninfo()
@quickactivate

using Plots, ProgressMeter, Logging, WeightsAndBiasLogger
theme(:bright; size=(300, 300))

using Random, Turing, BayesianSymbolic
using ExprOptimization.ExprRules

includef(args...) = isdefined(Main, :Revise) ? includet(args...) : include(args...)
includef(srcdir("utility.jl"))
includef(srcdir("app_inf.jl"))
includef(srcdir("sym_reg.jl"))
includef(srcdir("exp_max.jl"))
includef(srcdir("analyse.jl"))
includef(srcdir("dataset.jl"))
# Suppress warnings of using _varinfo
with_logger(SimpleLogger(stderr, Logging.Error)) do
    includef(srcdir("scenarios", "magnet.jl"))
    includef(srcdir("scenarios", "nbody.jl"))
    includef(srcdir("scenarios", "bounce.jl"))
    includef(srcdir("scenarios", "mat.jl"))
    includef(srcdir("scenarios", "spring.jl"))
    includef(srcdir("scenarios", "fall.jl"))
end

isdebugbsp() = "JULIA_DEBUG" in keys(ENV) && ENV["JULIA_DEBUG"] == "bsp"

const CE_PRESET = CrossEntropy(800, 2, 10, 600, 200.0)
const CE_DEBUG  = CrossEntropy( 20, 2, 10,  15,   5.0)

function loadpreset(dataset; priordim::Bool=true, priortrans::Bool=true)
    local grammar
    if priordim == true
        if priortrans == true
            grammar = G_BSP_11
        else
            grammar = G_BSP_10
        end
    else
        if priortrans == true
            grammar = G_BSP_01
        else
            grammar = G_BSP_00
        end
    end
    ce = isdebugbsp() ? CE_DEBUG : CE_PRESET
    preset = Dict(
        "synth/magnet" => (
            ScenarioModel=MagnetScenario, 
            latentname=["fric2", "magn3"],
        ),
        "synth/nbody" => (
            ScenarioModel=NBodyScenario, 
            latentname=["mass1", "mass2", "mass3"],
            ealg = ImportanceSampling(nsamples=200, rsamples=3),
            malg = BSPForce(grammar=grammar, opt=ce, beta=1e-3),
            elike = Likelihood(nahead=5, nlevel=0.01),
            mlike = Likelihood(nahead=1, nlevel=0.01),
        ),
        "synth/bounce" => (
            ScenarioModel=BounceScenario,
            latentname=["mass1", "mass2", "mass3", "mass4"],
            ealg = ImportanceSampling(nsamples=200, rsamples=3),
            malg = BSPForce(grammar=grammar, opt=ce, beta=1e-3),
            elike = Likelihood(nahead=5, nlevel=0.01),
            mlike = Likelihood(nahead=1, nlevel=0.01),
        ),
        "synth/mat" => (
            ScenarioModel=MatScenario, 
            latentname=["fric2", "magn3"],
            ealg = ImportanceSampling(nsamples=200, rsamples=3),
            malg = BSPForce(grammar=grammar, opt=ce, beta=1e-3),
            elike = Likelihood(nahead=5, nlevel=0.001),
            mlike = Likelihood(nahead=1, nlevel=0.001),
        ),
        "phys101/fall" => loadpreset_fall(),
        "phys101/spring" => loadpreset_spring(),
    )
    return preset[dataset]
end

@cast function eonly(
    dataset::String, ntrains::Int;
    shuffleseed::Int=-1,
    slient::Bool=false, 
)
    scenarios, attributes = loaddata(datadir(dataset), ntrains; shuffleseed=shuffleseed, verbose=!slient)
    @unpack ScenarioModel, latentname, ealg, malg, elike, mlike = loadpreset(dataset)
    
    # E-step with zero force
    latents_zero = estep(ealg, ScenarioModel, scenarios, ZeroForce(), elike; verbose=true)
    est_zero = expect.(x -> x, latents_zero)
    mse_zero = mean(map(d -> d[1]^2, est_zero - attributes))
    
    # E-step with oracle force
    latents_oracle = estep(ealg, ScenarioModel, scenarios, OracleForce(), elike; verbose=true)
    est_oracle = expect.(x -> x, latents_oracle)
    mse_oracle = mean(map(d -> d[1]^2, est_oracle - attributes))
    
    !slient && @info "E-step only" mse_zero mse_oracle
    return (zero=mse_zero, oracle=mse_oracle)
end

@cast function monly(
    dataset::String, ntrains::Int;
    seed::Int=0, shuffleseed::Int=-1, priordim::Bool=true, priortrans::Bool=true,
    slient::Bool=false, nosave::Bool=false, depthcount::Int=6,
)
    hps = @ntuple(ntrains, seed, shuffleseed, priordim, priortrans)

    scenarios, attributes = loaddata(datadir(dataset), ntrains; shuffleseed=shuffleseed, verbose=!slient)
    @unpack ScenarioModel, latentname, ealg, malg, elike, mlike =
        loadpreset(dataset; priordim=priordim, priortrans=priortrans)
    
    nexprs = count_expressions(malg.grammar, depthcount, :Force)
    !slient && @info "Num of expressions upto depth $depthcount" nexprs
    
    # M-step with oracle latent
    Random.seed!(seed)
    latents = make_latents(attributes) # orcale latents
    tused = @elapsed force = mstep(malg, ScenarioModel, scenarios, latents, mlike; verbose=true)
    
    expr = BayesianSymbolic.get_executable(force.tree, force.grammar)
    !slient && @info "M-step only" expr tused
    !nosave && wsave(
        resultsdir(dataset, savename(hps; connector="-"), "monly.jld2"), 
        @strdict(ScenarioModel, latentname, ealg, malg, elike, mlike, force)
    )
end

@cast function em(
    dataset::String, ntrains::Int, niters::Int; 
    seed::Int=0, shuffleseed::Int=-1,
    slient::Bool=false, logging::Bool=false, nosave::Bool=false, depthcount::Int=6,
)
    hps = @ntuple(ntrains, seed, shuffleseed)

    scenarios, attributes = loaddata(datadir(dataset), ntrains; shuffleseed=shuffleseed, verbose=!slient)
    @unpack ScenarioModel, latentname, ealg, malg, elike, mlike = loadpreset(dataset)
    
    nexprs = count_expressions(malg.grammar, depthcount, :Force)
    !slient && @info "Num of expressions upto depth $depthcount" nexprs
    
    logging && (logger = WBLogger(project="BSP"))
    logging && config!(logger, hps)
    
    # EM
    Random.seed!(seed)
    trace = []
    force = ZeroForce()
    slient && (progress = Progress(niters, "Exp-Max"))
    for iter in 1:niters
        !slient && @info "Exp-Max ($iter/$niters)"
        latents = estep(ealg, ScenarioModel, scenarios, force, elike; verbose=!slient)
        force = mstep(malg, ScenarioModel, scenarios, latents, mlike; verbose=!slient)        
        logging && with_logger(logger) do
            nrmse = compute_normrmse(ScenarioModel, scenarios, latents, force, mlike)
            @info "train" iter=iter normrmse=nrmse
        end
        slient && next!(progress)
        push!(trace, (latents=latents, force=force))
    end
    
    expr = BayesianSymbolic.get_executable(force.tree, force.grammar)
    !slient && @info "EM" expr force.constant
    !nosave && wsave(
        resultsdir(dataset, savename(hps; connector="-"), "em.jld2"), 
        @strdict(ScenarioModel, latentname, ealg, malg, elike, mlike, trace)
    )
end

#function get_init_state(traj, eps::AbstractFloat)
#    pos1, pos2 = traj[1].position, traj[2].position
#    return State(position=pos2, velocity=((pos2 - pos1) / eps))
#end

#function predict_trajectory(entity, state, getforce, eps::AbstractFloat, T::Int)
#    traj = [state]
#    for t in 1:T
#        push!(traj, run(Euler(eps), getforce, entity, traj[end]))
#    end
#    return traj
#end

#initial_state = get_init_state(traj, hps.eps)
#predicted_trajectory = predict_trajectory(
#    entity, initial_state, getforce, hps.eps, predictlen_max
#)

#short_trajectory = predicted_trajectory[1:predictlen_min]
#anim = @animate for t in 1:length(short_trajectory)
#    plot(entity, short_trajectory, t)
#end
#gif(anim, resultsdir(dataset, sn, "W1S$i-shortpred.gif"); show_msg=false)

#anim = @animate for t in 1:length(predicted_trajectory)
#    plot(entity, predicted_trajectory, t)
#end
#gif(anim, resultsdir(dataset, sn, "W1S$i-pred.gif"); show_msg=false)
    
@main
