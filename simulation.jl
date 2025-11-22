"""
Simulation functions for budget-constrained planning.
This module provides high-level functions to set up and run simulations.
"""

using Random
using StaticArrays
using Statistics
using PythonCall
using Revise

# Import local modules with Revise for automatic reloading
includet("src/fov.jl")
includet("src/rrt_star.jl")
includet("src/world.jl")
includet("src/visual_odometry.jl")
includet("src/nominal_planner.jl")
includet("src/backup_planner.jl")
includet("src/utils.jl")
includet("src/gatekeeper.jl")

using .World
using .VO

"""
    load_features_from_image(image_path::String, domain_size::Real; 
                            max_corners=1000, quality_level=0.04, min_distance=10)

Load corner features from an image using OpenCV's goodFeaturesToTrack.
Returns a BallTree of features in meter coordinates and the aspect ratio.
"""
function load_features_from_image(image_path::String, domain_size::Real; 
                                 max_corners=1000, quality_level=0.04, min_distance=10)
    cv2 = pyimport("cv2")
    image = cv2.imread(image_path)
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features_raw = cv2.goodFeaturesToTrack(gray_im, max_corners, quality_level, min_distance)
    cornersPlot = dropdims(pyconvert(Array{Int}, features_raw), dims=2)
    
    h, w = gray_im.shape
    AR = pyconvert(Float64, w) / pyconvert(Float64, h)
    
    # Convert pixel coordinates to meters
    corners = cornersPlot .* domain_size / pyconvert(Float64, h)
    # Flip y axis
    corners[:, 2] = domain_size .- corners[:, 2]
    
    features = BallTree(corners', Euclidean(), reorder=false)
    
    return features, AR
end

"""
    setup_simulation(;
        image_path="images/field.png",
        domain_size=200.0,
        seed=1337,
        goal_pos=[230.0, 175.0, 1.2*π/2],
        start_pos=[30.0, 130.0, -π/2],
        mid_landmarks=Vector{SVector{3,Float64}}(),
        unsafe_zones=World.AbstractUnsafeZone[],
        vo_params=nothing,
        max_cost=9.0,
        nominal_iterations=10000,
        backup_iterations=100
    )

Set up all components needed for a simulation run.

Returns a named tuple with:
- features: BallTree of map features
- AR: aspect ratio of the domain
- domain: domain bounds
- nominal_planner: initialized nominal planner
- backup_planner: initialized backup planner
- vo_params: visual odometry parameters
- x0: initial state
- goal: goal state
- max_cost: maximum allowable cost
- initial_traj: initial committed trajectory
"""
function setup_simulation(;
    image_path="images/field.png",
    domain_size=200.0,
    seed=1337,
    goal_pos=[230.0, 175.0, 1.2*π/2],
    start_pos=[30.0, 130.0, -π/2],
    mid_landmarks=Vector{SVector{3,Float64}}(),
    unsafe_zones=World.AbstractUnsafeZone[],
    vo_params=nothing,
    max_cost=9.0,
    nominal_iterations=10000,
    backup_iterations=100
)
    
    # Load features from image
    features, AR = load_features_from_image(image_path, domain_size)
    
    # Set up domain
    domain = (@SVector[0.0, 0.0, 0.0], @SVector[domain_size*AR, domain_size, 2*π])
    goal = SVector{3}(goal_pos)
    x0 = SVector{3}(start_pos)
    
    # Initialize nominal planner
    nominal_planner = initialize_nominal_planner(goal, domain, unsafe_zones, nominal_iterations)
    
    # Set up VO parameters if not provided
    if isnothing(vo_params)
        vo_params = VO.VOParams(
            fovRadius=60.0, 
            fovAngle=90*π/180, 
            errorRate=0.03, 
            updateRate=5.0, 
            minFeatures=8
        )
    end
    
    # Initialize seen features
    starting_features_idx = features_in_fov_idx(features, x0, vo_params.fovRadius, 2π)
    seen_features = BallTree(features.data[starting_features_idx], Euclidean(), reorder=false)
    
    # Set up landmarks
    landmarks = [x0, mid_landmarks..., goal]
    
    # Initialize backup planner
    backup_planner = initialize_backup_planner(
        domain,
        vo_params,
        landmarks,
        MVector(x0[1], x0[2], vo_params.fovRadius),
        SizedVector(seen_features),
        unsafe_zones,
        backup_iterations
    )
    
    # Get initial nominal path and trajectory
    _, initial_nom_path = query_nominal_planner(nominal_planner, x0)
    initial_nom_path = make_path_from_waypoints(initial_nom_path, nominal_planner.prob.turning_radius)
    
    initial_traj = CompositeTrajectory([], [], 0.0, 0.0, 0.0)
    new_committed, committed_traj = gatekeeper(x0, initial_traj, nominal_planner, backup_planner, max_cost)
    
    if !new_committed
        error("Failed to find initial committed trajectory")
    end
    
    return (
        features=features,
        AR=AR,
        domain=domain,
        domain_size=domain_size,
        nominal_planner=nominal_planner,
        backup_planner=backup_planner,
        vo_params=vo_params,
        x0=x0,
        goal=goal,
        max_cost=max_cost,
        initial_traj=committed_traj
    )
end

"""
    run_simulation(setup; max_iterations=800, update_backup_every=5, 
                   backup_max_nodes=4000, backup_grow_iterations=10, verbose=true,
                   use_controller=true, lookahead_dist=10.0)

Run the main simulation loop using the provided setup.

If use_controller=true, uses a tracking controller with Dubins dynamics.
Otherwise, samples trajectory directly (old behavior).

Returns a named tuple with complete simulation history:
- x_hist: state history
- gk_hist: gatekeeper/committed trajectory history
- features_hist: observed features history
- backup_hist: backup planner tree history
- cost_hist: cumulative cost history
- num_feats_hist: number of features seen history
- new_committed_hist: history of gatekeeper success
- gk_times: gatekeeper computation times
- reroot_times: backup planner update times
- t_f: final simulation time
- control_hist: control input history (if use_controller=true)
"""
function run_simulation(setup; 
                       max_iterations=800, 
                       update_backup_every=5, 
                       backup_max_nodes=4000,
                       backup_grow_iterations=10,
                       verbose=true,
                       use_controller=true,
                       lookahead_dist=10.0)
    
    # Unpack setup
    features = setup.features
    nominal_planner = setup.nominal_planner
    backup_planner = setup.backup_planner
    vo_params = setup.vo_params
    x0 = setup.x0
    max_cost = setup.max_cost
    committed_traj = setup.initial_traj
    
    # Simulation parameters
    ΔT = 1 / backup_planner.prob.vo_params.updateRate
    current_cost = 0.0
    
    # Initialize history
    x_hist = [x0]
    gk_hist = [committed_traj]
    features_hist = [backup_planner.prob.mapped_features[1]]
    backup_hist = [copy(backup_planner.nodes)]
    cost_hist = [current_cost]
    num_feats_hist = [NaN]
    new_committed_hist = [1.0]
    control_hist = SVector{2,Float64}[]
    
    x = x0
    t = t_commit = 0.0
    
    gk_times = Float64[]
    reroot_times = Float64[]
    
    # Main simulation loop
    for i in 1:max_iterations
        t += ΔT
        t_backup_reached = t_commit + committed_traj.switch_time + committed_traj.backup_time
        
        # Compute state update
        if use_controller
            # Get target waypoint from trajectory
            target = get_trajectory_target(committed_traj, t - t_commit, 
                                          nominal_planner.prob.max_velocity, lookahead_dist)
            
            # Compute control input
            u = waypoint_controller(x, target, nominal_planner.prob.max_velocity, 
                                   backup_planner.prob.turning_radius, lookahead_dist)
            push!(control_hist, u)
            
            # Propagate dynamics
            x = dubins_dynamics(x, u, ΔT, backup_planner.prob.turning_radius)
        else
            # Old behavior: sample trajectory directly
            x = sample_committed_trajectory(committed_traj, t - t_commit, nominal_planner.prob.max_velocity)
        end
        
        push!(x_hist, x)
        
        # Check if landmark is visible (reset cost if so)
        landmark_seen = false
        for l in backup_planner.prob.landmarks
            if t >= t_backup_reached || is_in_fov(x, l, backup_planner.prob.turning_radius*1.1, 2π) || is_in_fov(x, l, vo_params.fovRadius, vo_params.fovAngle)
                if verbose && i % 10 == 0
                    println("Iter $i: landmark seen")
                end
                current_cost = 0.0
                landmark_seen = true
                break
            end
        end
        
        # Update odometry error
        if !landmark_seen
            cost, num_feats = VO.odometry_error(x_hist[end-1], x, features, vo_params)
            current_cost += cost
        else
            num_feats = NaN
        end
        push!(num_feats_hist, num_feats)
        push!(cost_hist, current_cost)
        
        if verbose && i % 10 == 0
            println("Iter $i: x: $x, cost: $current_cost")
        end
        
        # Check if goal reached
        goal_reached = false
        if norm(x[SOneTo(2)] - backup_planner.prob.landmarks[end]) <= backup_planner.prob.turning_radius
            if verbose
                println("Iter $i: goal reached")
            end
            goal_reached = true
        end
        
        # Update backup planner
        reroot_time = @elapsed begin
            new_features_idx = features_in_fov_idx(features, x, vo_params.fovRadius, vo_params.fovAngle)
            if i % update_backup_every == 0 && length(backup_planner.nodes) < backup_max_nodes
                update_backup_planner!(
                    backup_planner,
                    MVector(x[1], x[2], vo_params.fovRadius * 2),
                    Set(features.data[new_features_idx]),
                    backup_grow_iterations
                )
            end
        end
        push!(reroot_times, reroot_time)
        push!(features_hist, backup_planner.prob.mapped_features[1])
        push!(backup_hist, copy(backup_planner.nodes))
        
        # Run gatekeeper
        gk_time = @elapsed begin
            new_committed, committed_traj = gatekeeper(
                x, committed_traj, nominal_planner, backup_planner, max_cost - current_cost
            )
        end
        push!(gk_times, gk_time)
        
        if new_committed
            if verbose && i % 10 == 0
                println("Iter $i: gk finished w/ cost: ", committed_traj.total_cost, 
                       " and switch_time: ", committed_traj.switch_time)
            end
            t_commit = t
        else
            if verbose && i % 10 == 0
                println("Iter $i: gk failed")
            end
        end
        push!(gk_hist, committed_traj)
        push!(new_committed_hist, new_committed ? 1 : NaN)
        
        # Check termination conditions
        if current_cost > max_cost
            if verbose
                println("Iter $i: max cost reached, cost = $current_cost")
            end
            break
        end
        if goal_reached
            break
        end
    end
    
    t_f = t
    
    return (
        x_hist=x_hist,
        gk_hist=gk_hist,
        features_hist=features_hist,
        backup_hist=backup_hist,
        cost_hist=cost_hist,
        num_feats_hist=num_feats_hist,
        new_committed_hist=new_committed_hist,
        gk_times=gk_times,
        reroot_times=reroot_times,
        t_f=t_f,
        ΔT=ΔT,
        control_hist=control_hist
    )
end

"""
    compute_timing_stats(sim_results)

Compute mean and standard deviation of timing data from simulation results.
Returns (mean_reroot, std_reroot, mean_gk, std_gk) in milliseconds.
"""
function compute_timing_stats(sim_results)
    mean_reroot = mean(sim_results.reroot_times) * 1000
    std_reroot = std(sim_results.reroot_times) * 1000
    mean_gk = mean(sim_results.gk_times) * 1000
    std_gk = std(sim_results.gk_times) * 1000
    
    return (mean_reroot=mean_reroot, std_reroot=std_reroot, 
            mean_gk=mean_gk, std_gk=std_gk)
end

"""
    setup_omniscient_planner(setup; iterations=3000)

Create an omniscient backup planner that has access to all features.
Useful for computing optimal trajectories for comparison.
"""
function setup_omniscient_planner(setup; iterations=3000)
    Random.seed!(1337)
    
    omniscient = initialize_backup_planner(
        setup.domain,
        setup.vo_params,
        [setup.goal],
        MVector(100.0, 100.0, 150.0),
        SizedVector(setup.features),
        setup.nominal_planner.prob.unsafe_zones,
        iterations
    )
    
    return omniscient
end

"""
    compute_omniscient_cost(omniscient, x0)

Compute the omniscient optimal cost and path from a starting position.
"""
function compute_omniscient_cost(omniscient, x0)
    cost, path = query_backup_planner(omniscient, x0)
    path = make_path_from_waypoints(path, omniscient.prob.turning_radius)
    return cost, path
end

"""
    compute_nominal_failure(nominal_planner, initial_nom_path, vo_params)

Compute where the nominal trajectory would fail and its total cost.
Returns (nominal_cost, failure_state, failure_index).
"""
function compute_nominal_failure(nominal_planner, initial_nom_path, vo_params)
    nom_cost = compute_nominal_trajectory_cost(nominal_planner.prob, initial_nom_path, Inf)
    nom_fail_idx = length(nom_cost)
    x_nom_fail = sample_combined_dubins_path(
        initial_nom_path, 
        nom_fail_idx * nominal_planner.prob.max_velocity / vo_params.updateRate
    )
    
    return nom_cost[end], x_nom_fail, nom_fail_idx
end

"""
    find_destination_index(nodes::Vector{RRTStar.Node{TV}}, start::Int64) where TV

Helper function to traverse backup planner tree to find destination node.
Used for visualization.
"""
function find_destination_index(nodes::Vector{RRTStar.Node{TV}}, start::Int64) where TV
    if nodes[start].parent_index == 0
        return start
    else
        return find_destination_index(nodes, nodes[start].parent_index)
    end
end
