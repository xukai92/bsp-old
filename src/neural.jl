# Neural

abstract type NeuralModel end
abstract type NeuralForceModel <: NeuralModel end
abstract type NeuralDynamicsModel <: NeuralModel end

### Hacks for neural dynamics

using BayesianSymbolic: AbstractDynamics

function BayesianSymbolic.run(
    ::AbstractDynamics, ndm::NeuralDynamicsModel, entity::Entity, state::State, constraint::T=nothing
) where {T}
    position, velocity = ndm(entity, state)
    return State(position=position, velocity=velocity)
end

function make_shape_vec(shape)
    # First two digits are binary encoding for type of shapes
    # Disc: 00
    # RectWall: 01
    # RectMat: 10
    if shape isa Disc
        @unpack radius = shape
        return [0, 0, radius, radius]
    elseif shape isa RectWall
        @unpack width, height = shape
        return [0, 1, width,  height]
    elseif shape isa RectMat
        @unpack width, height = shape
        return [1, 0, width,  height]
    end
end

make_shape_vec(ett::Entity) = make_shape_vec(ett.shape)

# OGN force

struct OGNForceModel <: NeuralForceModel
    nn_node
    nn_edge
    losses_train
end

OGNForceModel(nn_node, nn_edge) = OGNForceModel(nn_node, nn_edge, [])

Flux.params(nfm::OGNForceModel) = Flux.params(nfm.nn_node, nfm.nn_edge)

function TrackerFlux.track(nfm::OGNForceModel)
    return OGNForceModel(
        TrackerFlux.track(nfm.nn_node), 
        TrackerFlux.track(nfm.nn_edge), 
        nfm.losses_train,
    )
end

function TrackerFlux.untrack(nfm::OGNForceModel)
    return OGNForceModel(
        TrackerFlux.untrack(nfm.nn_node), 
        TrackerFlux.untrack(nfm.nn_edge), 
        nfm.losses_train,
    )
end

function gen_getforce(nfm::OGNForceModel; mass_scale=MASS_SCALE)
    function getforce_pair(etti::Entity, si::State, ettj::Entity, sj::State)
        c = contact_point(etti, si, ettj, sj)
        mi = etti.mass' * mass_scale
        mj = ettj.mass' * mass_scale
        ui = etti.friction'
        uj = ettj.friction'
        position_mean = (si.position + sj.position) / 2
        velocity_mean = (si.velocity + sj.velocity) / 2
        shapei = repeat(make_shape_vec(etti), 1, size(position_mean, 2))
        shapej = repeat(make_shape_vec(ettj), 1, size(position_mean, 2))
        embdi = cat(mi, ui, shapei, c, si.position - position_mean, si.velocity - velocity_mean; dims=1) |> nfm.nn_node
        embdj = cat(mj, uj, shapej, c, sj.position - position_mean, sj.velocity - velocity_mean; dims=1) |> nfm.nn_node
        acc = cat(embdi, embdj; dims=1) |> nfm.nn_edge
        return acc .* etti.mass'
    end
    return (e, s) -> getforce_pairwise(e, s, getforce_pair)
end

# MLP force

struct MLPForceModel <: NeuralForceModel
    nn
    losses_train
end

MLPForceModel(nn) = MLPForceModel(nn, [])

Flux.params(nfm::MLPForceModel) = Flux.params(nfm.nn)

function TrackerFlux.track(nfm::MLPForceModel)
    return MLPForceModel(
        TrackerFlux.track(nfm.nn), 
        nfm.losses_train,
    )
end

function TrackerFlux.untrack(nfm::MLPForceModel)
    return MLPForceModel(
        TrackerFlux.untrack(nfm.nn), 
        nfm.losses_train,
    )
end

function gen_getforce(nfm::MLPForceModel; mass_scale=MASS_SCALE)
    function getforce_pair(etti::Entity, si::State, ettj::Entity, sj::State)
        c = contact_point(etti, si, ettj, sj)
        mi = etti.mass' * mass_scale
        mj = ettj.mass' * mass_scale
        ui = etti.friction'
        uj = ettj.friction'
        position_mean = (si.position + sj.position) / 2
        velocity_mean = (si.velocity + sj.velocity) / 2
        shapei = repeat(make_shape_vec(etti), 1, size(position_mean, 2))
        shapej = repeat(make_shape_vec(ettj), 1, size(position_mean, 2))
        inpi = cat(mi, ui, shapei, c, si.position - position_mean, si.velocity - velocity_mean; dims=1)
        inpj = cat(mj, uj, shapej, c, sj.position - position_mean, sj.velocity - velocity_mean; dims=1)
        return nfm.nn(cat(inpi, inpj; dims=1)) .* etti.mass'
    end
    return (e, s) -> getforce_pairwise(e, s, getforce_pair)
end

function m_step!(nm::NeuralModel, scenes, n_total_passes; batch_on=false, lr=5f-3, lrdecay_on=false)
    loss_cache = Ref(0.0)
    getforce = nm isa NeuralForceModel ? gen_getforce(nm) : nm
    function lossf(scenes; eval_only=false)
        loss = get_likelihood(scenes, getforce; get_logjoint=get_logjoint_tracker)
        loss_cache[] += loss |> TrackerFlux.untrack
        if eval_only
            return loss |> TrackerFlux.untrack
        else
            return loss
        end
    end
    
    ps = Flux.params(nm)
    opt = lrdecay_on ? Flux.Optimiser(ExpDecay(), ADAM(lr)) : ADAM(lr)

    push!(nm.losses_train, (0, lossf(scenes; eval_only=true)))

    n_scenes_train = length(scenes)
    n_epoches = div(n_total_passes, n_scenes_train)
    progress = Progress(n_epoches)
    for i_epoch in 1:n_epoches
        loss_cache[] = 0.0
        data = batch_on ? [[scenes[i:i]] for i in shuffle(1:n_scenes_train)] : [[scenes]]
        Flux.train!(lossf, ps, data, opt)
        push!(nm.losses_train, (i_epoch * n_scenes_train, loss_cache[] / length(data)))
        ProgressMeter.next!(progress; showvalues = [(:i_epoch, i_epoch), (:loss, nm.losses_train[end][2])])
    end
    
    return plot(
        map(l -> l[1], nm.losses_train), map(l -> l[2], nm.losses_train); 
        label=nothing, xlabel="Epoches", ylabel="Loss", size=(600, 300)
    )
end

# MLP dynamics

struct MLPDynamicsModel <: NeuralDynamicsModel
    nn
    losses_train
end

MLPDynamicsModel(nn) = MLPDynamicsModel(nn, [])

Flux.params(ndm::MLPDynamicsModel) = Flux.params(ndm.nn)

function TrackerFlux.track(ndm::MLPDynamicsModel)
    return MLPDynamicsModel(
        TrackerFlux.track(ndm.nn), 
        ndm.losses_train,
    )
end

function TrackerFlux.untrack(ndm::MLPDynamicsModel)
    return MLPDynamicsModel(
        TrackerFlux.untrack(ndm.nn), 
        ndm.losses_train,
    )
end

function (ndm::MLPDynamicsModel)(entity, state; mass_scale=MASS_SCALE)
    @unpack dynamic, mass, friction, shape = entity
    @unpack position, velocity = state
    dynamic = repeat(dynamic, 1, size(position, 3))
    mass = repeat(mass, 1, size(position, 3)) * mass_scale
    friction = repeat(friction, 1, size(position, 3))
    shape = repeat(cat(make_shape_vec.(shape)...; dims=1), 1, size(position, 3))
    inp = cat(dynamic, mass, friction, shape, position[1,:,:], position[2,:,:], velocity[1,:,:], velocity[2,:,:]; dims=1)
    pos_vel = ndm.nn(inp)
    dims_pos = prod(size(position)[1:2])
    position += reshape(pos_vel[1:dims_pos,:], size(position)...)
    velocity += reshape(pos_vel[dims_pos+1:end,:], size(velocity)...)
    return position, velocity
end

# IN

struct INDynamicsModel <: NeuralDynamicsModel
    nn_node
    nn_edge
    nn_transit
    losses_train
end

INDynamicsModel(nn_node, nn_edge, nn_transit) = INDynamicsModel(nn_node, nn_edge, nn_transit, [])

Flux.params(ndm::INDynamicsModel) = Flux.params(ndm.nn_node, ndm.nn_edge, ndm.nn_transit)

function TrackerFlux.track(ndm::INDynamicsModel)
    return INDynamicsModel(
        TrackerFlux.track(ndm.nn_node), 
        TrackerFlux.track(ndm.nn_edge), 
        TrackerFlux.track(ndm.nn_transit), 
        ndm.losses_train,
    )
end

function TrackerFlux.untrack(ndm::INDynamicsModel)
    return INDynamicsModel(
        TrackerFlux.untrack(ndm.nn_node), 
        TrackerFlux.untrack(ndm.nn_edge), 
        TrackerFlux.untrack(ndm.nn_transit), 
        ndm.losses_train,
    )
end

function (ndm::INDynamicsModel)(entity, state; mass_scale=MASS_SCALE)
    function getforce_pair(etti::Entity, si::State, ettj::Entity, sj::State)
        c = contact_point(etti, si, ettj, sj)
        mi = etti.mass' * mass_scale
        mj = ettj.mass' * mass_scale
        ui = etti.friction'
        uj = ettj.friction'
        di = etti.dynamic'
        dj = ettj.dynamic'
        position_mean = (si.position + sj.position) / 2
        velocity_mean = (si.velocity + sj.velocity) / 2
        shapei = repeat(make_shape_vec(etti), 1, size(position_mean, 2))
        shapej = repeat(make_shape_vec(ettj), 1, size(position_mean, 2))
        embdi = cat(mi, ui, di, shapei, c, si.position - position_mean, si.velocity - velocity_mean; dims=1) |> ndm.nn_node
        embdj = cat(mj, uj, dj, shapej, c, sj.position - position_mean, sj.velocity - velocity_mean; dims=1) |> ndm.nn_node
        acc = cat(embdi, embdj; dims=1) |> ndm.nn_edge
        return acc
    end
    @unpack position, velocity = state
    interaction = getforce_pairwise(entity, state, getforce_pair; d=size(ndm.nn_edge.layers[end].W, 1))
    pos_vel = ndm.nn_transit(cat([interaction[:,i,:] for i in 1:size(interaction, 2)]...; dims=1))
    dims_pos = prod(size(position)[1:2])
    position += reshape(pos_vel[1:dims_pos,:], size(position)...)
    velocity += reshape(pos_vel[dims_pos+1:end,:], size(velocity)...)
    return position, velocity
end

###

function evaluate_neural(nm::NeuralModel, scenes)
    nm = TrackerFlux.untrack(nm)
    getforce = nm isa NeuralForceModel ? gen_getforce(nm) : nm
    return evaluate(getforce, scenes)
end
