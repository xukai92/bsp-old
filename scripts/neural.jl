using DrWatson, InteractiveUtils
@quickactivate
versioninfo()

###

# using FileIO, Images, StatsPlots, Parameters, LaTeXStrings
# using LinearAlgebra, StatsFuns, Distributions

using Comonicon
using Logging, Printf, BSON, ProgressMeter
using Random, Optim, Turing, ExprOptimization, TrackerFlux, Flux, BayesianSymbolic
using Plots; theme(:bright; size=(300, 300)) 
TrackerFlux.overload_gradient()

###

# Global simulation arguments
const ARGS_SIM = (
    ε = 2e-2,   # step size of integrator
    L = 50,     # #{discretization steps}
    σ = 1e-2,   # observation noise level
)

const PASSES_PER_DATA = 2_000

###

function rmse_of(pos_1, pos_2; normalize=true)
    @assert size(pos_1) == size(pos_2)
    if ndims(pos_1) == 3
        _, N, L = size(pos_1)
    else
        (_, N), L = size(pos_1), 1
    end
    mse = sum((pos_1 - pos_2).^2)
    if normalize
        mse /= (N * L)
    end
    return sqrt(mse)
end

const DATASET = ENV["DATASET"]

if DATASET == "NBODY"
    BETA = 1e-0
    CONSTANT_SCALE = 1 / BayesianSymbolic.G * 1e6
    MASS_SCALE = sqrt(BayesianSymbolic.G) / 1e9
    include(scriptsdir("scenes/nbody.jl"))
elseif DATASET == "BOUNCE"
    BETA = 1e-1
    CONSTANT_SCALE = 1
    MASS_SCALE = 1
    include(scriptsdir("scenes/bounce.jl"))
elseif DATASET == "MAT"
    BETA = 1e-4
    CONSTANT_SCALE = 1
    MASS_SCALE = 1
    include(scriptsdir("scenes/mat.jl"))
end

###

include("force_models.jl")

function get_likelihood(scenes, getforce; get_logjoint=get_logjoint, is_seq=false, is_rmse=false)
    nlp_likelihood = 0
    for scene in scenes
        samples, data = scene
        nlp_likelihood_scene = 0
        for sample in samples
            m = condition_scene(sample, data, getforce; is_seq=is_seq, is_rmse=is_rmse)
            nlp_likelihood_scene += -get_logjoint(m)
        end
        nlp_likelihood += nlp_likelihood_scene / length(samples)
    end
    nlp_likelihood /= length(scenes)
    return nlp_likelihood
end

function evaluate(getforce, scenes)
    rmse_vec = get_likelihood(scenes, getforce; is_seq=false, is_rmse=true)
    rmse_seq = get_likelihood(scenes, getforce; is_seq=true,  is_rmse=true)
    return rmse_vec, rmse_seq
end

###

function prepare_scene(sim_all, t, i_scene)
    sim = sim_all[t][i_scene]
    entity, data = sim.entity, (sim.state0, sim.positions, sim.velocitys)
    return (samples=[entity2sample(entity)], data=data)
end

function make_subdata(sim_all, n_data, data_type; seed=1)
    rng = MersenneTwister(seed)
    n_scenes = length(sim_all[data_type])
    n_scenes_train = floor(Int, 0.8 * n_scenes)
    idcs_train, idcs_test = shuffle(rng, 1:n_scenes_train)[1:n_data], n_scenes_train+1:n_scenes
    [prepare_scene(sim_all, data_type, i) for i in idcs_train], [prepare_scene(sim_all, data_type, i) for i in idcs_test]
end

"Quick check of E-step"
function check_e_step(dataset, samples_test, data_test)
    @info "Unkown mass" only(samples_test).mass

    chain = with_logger(NullLogger()) do
        m = condition_scene((only(samples_test)..., mass=missing), data_test, gen_getforce(TrueForce()); is_seq=true)
        @info sample(m, NUTS(0.5), 500)
    end

    if dataset == "MAT"
        @info "Unkown friction" only(samples_test).friction

        chain = with_logger(NullLogger()) do
            m = condition_scene((only(samples_test)..., friction=missing), data_test, gen_getforce(TrueForce()); is_seq=true)
            @info sample(m, NUTS(0.5), 500)
        end
    end
end

sim_all, vis_all, data_type = load_data()

n_display_cols = 20
vis_grid = reshape([vis_all[t][i] for t in [data_type], i in 1:100], :, n_display_cols)

samples_test, data_test = prepare_scene(sim_all, data_type, 1)

let sample_test = only(samples_test)
    m = condition_scene(sample_test, data_test, gen_getforce(TrueForce()))
    @info get_logjoint(m)
end

check_e_step(DATASET, samples_test, data_test)

let (_, scenes_test) = make_subdata(sim_all, 0, data_type; seed=1)
    rmse_true_vec, rmse_true_seq = evaluate(gen_getforce(TrueForce()), scenes_test)
    rmse_zero_vec, rmse_zero_seq = evaluate(gen_getforce(ZeroForce()), scenes_test)

    @info "Refs" rmse_true_seq rmse_true_vec rmse_zero_seq rmse_zero_vec
end

###

@cast function data_effiency(grammar::String="bsp", neural_model::String=""; dosave::Bool=true)
    println("Running data_effiency on $DATASET with (grammar=$grammar, neural_model=$neural_model).")
    grammar_choices = ("bsp", "naive", "naive2")
    if grammar in grammar_choices
        G = grammar == "bsp" ? Gf : (grammar == "naive" ? Gf_naive : Gf_naive2)

        # Only report for BSP grammar; the naive one is too large to compute
        if grammar == "bsp"
            let depth = 5
                @info "Number of possible expressions up to depth $depth" count_expressions(G, depth, :Force)
            end 
        end

        sfm = let n_pop= 1_000, n_iters = 4, max_depth = 9, top_k = 500
            SymbolicForceModel(G, CrossEntropy(n_pop, n_iters, max_depth, top_k))
        end

        results_symbolic = Dict()
        for n_data in 1:10
            results_symbolic[n_data] = []
            for seed in 1:10
                scenes_train, scenes_test = make_subdata(sim_all, n_data, data_type; seed=seed)

                @time Tf, loss_f = m_step(sfm, scenes_train; n_trials=1)

                local constants
                getf_infer = let find_constants = gen_lossf(scenes_train; return_constants=true)
                    constants = find_constants(Tf, sfm.G; n_iters=3, verbose=true)
                    gen_getforce(Tf, sfm.G, constants)
                end
                rmse_seq = get_likelihood(scenes_test, getf_infer; is_seq=true,  is_rmse=true)
                rmse_vec = get_likelihood(scenes_test, getf_infer; is_seq=false, is_rmse=true)

                @info "Symbolic (M-step)" rmse_seq rmse_vec
                
                res = (vec=rmse_vec, seq=rmse_seq, tree=Tf, constants=constants)
                sn = grammar == "bsp" ? "symbolic" : (grammar == "naive" ? "naive_grammar" : "naive_grammar2")
                dosave && safesave(datadir("data_effiency", DATASET, data_type, sn, "n_data=$n_data-seed=$seed.bson"), ntuple2dict(res))
                push!(results_symbolic[n_data], res)
            end
        end

        let res = results_symbolic[5][end]
            print(res.constants)
            print_tree(res.tree, G)
        end
    else
        @warn "Skipping symbolic experiments as grammar=$grammar. Valid options are from $grammar_choices."
    end

    if neural_model != ""
        function init_ogn(n_inp_units=12, n_emb_units=50, n_hid_units=100, n_out_units=2, actf=relu)
            return OGNForceModel(
                Chain(
                    Dense(1 * n_inp_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_emb_units, actf),
                ),
                Chain(
                    Dense(2 * n_emb_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_out_units)
                ),
            ), 5f-3
        end

        function init_mlpforce(n_inp_units=12, n_hid_units=100, n_out_units=2, actf=relu)
            return MLPForceModel(
                Chain(
                    Dense(2 * n_inp_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_out_units)
                ),
            ), 1f-3
        end

        function init_mlpdynamics(n_inp_units=11, n_hid_units=100, n_out_units=2, actf=relu)
            N = only(samples_test).N
            return MLPDynamicsModel(
                Chain(
                    Dense(N * n_inp_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, 2 * N * n_out_units)
                ),
            ), 5f-4
        end

        function init_in(n_inp_units=13, n_emb_units=50, n_hid_units=100, n_out_units=2, actf=relu)
            N = only(samples_test).N
            return INDynamicsModel(
                Chain(
                    Dense(1 * n_inp_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_emb_units, actf),
                ),
                Chain(
                    Dense(2 * n_emb_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_emb_units)
                ),
                Chain(
                    Dense(N * n_emb_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, n_hid_units, actf),
                    Dense(1 * n_hid_units, 2 * N * n_out_units)
                ),
            ), 1f-3
        end

        results_neural = Dict()
        for n_data in 1:10
            results_neural[n_data] = []
            for seed in 1:10
                let sd = datadir("data_effiency", DATASET, data_type, neural_model, "n_data=$n_data-seed=$seed")
                    if isfile("$sd.bson") 
                        println("Skipping (n_data=$n_data, seed=$seed) ...")
                        continue
                    end
                    if neural_model == "ogn"
                        nm, lr = init_ogn()
                    elseif neural_model == "in"
                        nm, lr = init_in()
                    elseif neural_model == "mlpforce"
                        nm, lr = init_mlpforce()
                    elseif neural_model == "mlpdynamics"
                        nm, lr = init_mlpdynamics()
                    end
                    nm = nm |> TrackerFlux.track

                    scenes_train, scenes_test = make_subdata(sim_all, n_data, data_type; seed=seed)

                    @time p = m_step!(nm, scenes_train, n_data * PASSES_PER_DATA; lr=lr, batch_on=false, lrdecay_on=false)
                    
                    rmse_vec, rmse_seq = evaluate_neural(nm, scenes_test)

                    @info "OGN" nm.losses_train[end] rmse_vec rmse_seq
                    
                    res = (vec_train=nm.losses_train[end], vec=rmse_vec, seq=rmse_seq)
                
                    dosave && safesave("$sd.bson", ntuple2dict(res))
                    dosave && savefig(p, "$sd.png")
                    push!(results_neural[n_data], res)
                end
            end
        end

    end

end

###

function get_inference_setup(; n_samples=150)
    n_adapt = 150
    target_ratio = 0.75 
    max_depth = 4
    alg = NUTS(n_adapt, target_ratio; max_depth=max_depth)
    return alg, n_adapt + n_samples
end

function get_posterior_chain(sample_true, data, getf; progress=false, is_seq=false, kwargs...)
    sample_true = (sample_true..., friction=missing, friction_prior=sample_true.friction)
    m = condition_scene(sample_true, data, getf; is_seq=is_seq)
    alg, n_total = get_inference_setup(; kwargs...)
    return with_logger(NullLogger()) do
        sample(m, alg, n_total; progress=progress)
    end
end

"Drop chains with smallest ESS"
function remove_poors(chain, n_removes)
    n_samples, dim, n_chains = size(chain)
    ESSs = map(1:n_chains) do i
        ss = summarystats(chain[:,:,i])
        mean(ss.nt.ess)
    end
    idcs = sortperm(ESSs)
    return chain[:,:,idcs[n_removes+1:end]]
end

"Fetch the last sample from each chain as a named tuple"
function fetch_samples(chain)
    N = 4
    n_chains = size(chain, 3)
    friction_raw = get(chain, :friction).friction
    friction_raw = [friction_raw]
    return map(1:n_chains) do i_chain
        friction = [map(j -> friction_raw[j][end,i_chain], 1:1)...]
        @ntuple(friction)
    end
end

function get_posterior_sample(scene, sample_size, getforce_candidate; n_removes=2, kwargs...)
    samples_true, data = scene
    chains = Vector{Any}(undef, sample_size + n_removes)
    for i in eachindex(chains)
        print(".")
        chains[i] = get_posterior_chain(only(samples_true), data, getforce_candidate; kwargs...)
    end
    chain = cat(chains...; dims=3)
    if n_removes > 0
        chain = remove_poors(chain, n_removes)
    end
    samples = fetch_samples(chain)
    samples = map(samples) do s
        (only(samples_true)..., friction=s.friction)
    end
    lps = [get(chain, [:lp]).lp[end,:]...]
    return @ntuple(samples, lps, chain)
end

function e_step(scenes, sample_size, getforce_candidate; kwargs...)
    lp_mat = zeros(sample_size, length(scenes))
    chains = Vector{Any}(undef, length(scenes))
    print("Running HMC for scene: ")
    samples = map(enumerate(scenes)) do (i, scene)
        print("$i")
        _, data = scene
        samples, lps, chain = get_posterior_sample(scene, sample_size, getforce_candidate; kwargs...)
        lp_mat[:,i] .= lps
        chains[i] = chain
        @ntuple(samples, data)
    end
    println()
    return samples, vec(mean(lp_mat; dims=2)), chains
end

function get_regression_setup()
    n_pop = 1_000
    n_iters = 4
    max_depth = 8
    top_k = 500
    alg = CrossEntropy(n_pop, n_iters, max_depth, top_k)
    return alg
end

function m_step(grammar, scenes, n_repeats::Int)
    alg = get_regression_setup()
    sfm = SymbolicForceModel(grammar, alg)
    return m_step(sfm, scenes; n_trials=n_repeats)
end

###

function em!(
    dataset_train, dataset_test, grammar, n_iters, sample_size, n_repeats; 
    getforce_init=gen_getforce(ZeroForce()), trace=[]
)
    getforce_candidate = getforce_init  # initialize force function
    println("EM starts")
    println()
    for iter in 1:n_iters
        println("Running EM iteration $iter")
        # E-step
        println("Running E-step")
        t_s = @elapsed samples, lps, chains = e_step(dataset_train, sample_size, getforce_candidate)
        lps_str = join(["[", "]"], join([@sprintf("%.3f", i) for i in lps], ", "))
        @printf("  (E-step, %6.3f s) lps = %s\n", t_s, lps_str)
        # M-step
        println("Running M-step")
        t_m = @elapsed tree, loss_m = m_step(grammar, samples, n_repeats)
        # Update force function
        local constants
        getforce_candidate = let find_constants = gen_lossf(samples; return_constants=true)
            constants = find_constants(tree, grammar; n_iters=3, verbose=true)
            gen_getforce(tree, grammar, constants)
        end
        rmse_seq = get_likelihood(dataset_test, getforce_candidate; is_seq=true,  is_rmse=true)
        rmse_vec = get_likelihood(dataset_test, getforce_candidate; is_seq=false, is_rmse=true)
        @printf("  (M-step, %6.3f s) loss = %.3f, seq = %.3f, vec = %.3f\n", t_m, loss_m, rmse_seq, rmse_vec)
        # Track
        @printf("Iter %2d done\n\n", iter)
        push!(trace, @ntuple(samples, lps, chains, tree, constants, loss_m, rmse_seq, rmse_vec))
    end
    println("EM ends")
    return trace
end


@cast function exp_max(dataset::String, symbolic_on::Bool, neural_on::Bool)

    scenes_train, scenes_test = let n_data = 5, seed = 1
        make_subdata(sim_all, n_data; seed=seed)
    end

    for scene in scenes_train
        samples, data = scene
        println(only(samples))
    end

    trace = []
    let n_iters = 5, sample_size = 2, n_repeats = 2
        em!(scenes_train, scenes_test, Gf, n_iters, sample_size, n_repeats; trace=trace)
        print_tree(trace[end].tree, Gf)
    #     safesave(
    #         datadir("exp_2", expname, "results-em", "n_data=3-seed=2.bson"), :trace=>trace
    #     )
    end

    pgfplotsx()
    theme(:default)

end

@main name="run-exps"
