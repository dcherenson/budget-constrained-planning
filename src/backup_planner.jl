# include("rrt_star.jl")
# include("vo.jl")

using .RRTStar
using .World
using StaticArrays
using Dubins
using NearestNeighbors
using .VO

# get just the distance between two nodes
function dubins_distance(q1, q2, turning_radius=0.1)

    e, p = Dubins.dubins_shortest_path(q1, q2, turning_radius, 1e-3)
    @assert e == Dubins.EDUBOK

    return Dubins.dubins_path_length(p)
end

@kwdef struct BackupPlannerProblem{F, VW, TT} <: RRTStar.AbstractProblem{SVector{3,F}}
    domain::Tuple{SVector{3,F}, SVector{3,F}} # rectangle defined by opposite corners
    pos_sampling_space::MVector{3,F} # x-y center and radius
    landmarks::Vector{SVector{2,F}}
    yaw_domain::Tuple{F, F} = (0.0, 2pi)# min and max yaw
    turning_radius::F = 10.0
    max_velocity::F = 10.0
    near_radius::F = 20.0
    mapped_features::SizedVector{TT}
    vo_params::VO.VOParams = VO.VOParams()
    unsafe_zones::VW
    root_nodes::Vector{Int} = Int[]
end

struct BackupPlanner{TP, TN}
    prob::TP
    nodes::TN
end

function initialize_backup_planner(domain::Tuple{TV, TV}, vo_params::VO.VOParams, landmarks::Vector{TV}, sample_space::MVector{3,F}, features::TF, zones::Vector{W}, max_iter) where {F, TV, TF, W<: World.AbstractUnsafeZone}
    P = BackupPlannerProblem(domain = domain,
                landmarks = SVector{2,F}[],
                vo_params = vo_params,
                unsafe_zones = zones,
                mapped_features = features,
                pos_sampling_space = sample_space)
    nodes = Vector{RRTStar.Node{TV}}()
    planner = BackupPlanner(P, nodes)
    for l in landmarks
        center = l[SOneTo(2)] + planner.prob.turning_radius*@SVector[-sin(l[3]), cos(l[3])]
        push!(planner.prob.landmarks, center)
    end

    activate_landmark!(planner, 1)

    RRTStar.rrt_star!(P, nodes, max_iter; do_rewire = true)
    return planner
end

function activate_landmark!(planner::BackupPlanner, idx::Int)
    center = planner.prob.landmarks[idx]
    for angle = 0.0:pi/2:3pi/2
        x = center[1] + planner.prob.turning_radius * cos(angle)
        y = center[2] + planner.prob.turning_radius * sin(angle)
        yaw = angle_diff(angle, pi/2)
        flipped_goal = @SVector[x, y, yaw]
        push!(planner.nodes, RRTStar.Node(flipped_goal))
        push!(planner.prob.root_nodes, length(planner.nodes))
    end
end

function update_backup_planner!(planner::BackupPlanner, new_fov::TS, new_features::Set{TV}, max_iter) where {TV, TS}
    union!(new_features, Set(planner.prob.mapped_features[1].data))
    planner.prob.mapped_features[1] = BallTree(collect(new_features), Euclidean(), reorder=false)
    planner.prob.pos_sampling_space .= new_fov
    RRTStar.rrt_star!(planner.prob, planner.nodes, max_iter; do_rewire = true)
end

function query_backup_planner(planner::BackupPlanner, start)
    start_flipped = @SVector [start[1], start[2], angle_add(start[3], pi)]
    cost, path = RRTStar.get_best_path(planner.prob, planner.nodes, start_flipped, rev=false)
    
    if !isfinite(cost)
        return Inf, path
    end

    for i in 1:length(path)
        path[i] = SVector(path[i][1], path[i][2], angle_add(path[i][3], pi))
    end
    return cost, path
end

function sample_domain(P::BackupPlannerProblem)
    v = @SVector rand(3)
    r = P.pos_sampling_space[3] * sqrt(v[1])
    theta = 2pi * v[2]
    yaw = P.yaw_domain[1] + v[3] * (P.yaw_domain[2] - P.yaw_domain[1])
    return SVector(P.pos_sampling_space[1] + r * cos(theta), P.pos_sampling_space[2] + r * sin(theta), yaw)
end

function RRTStar.sample_free(problem::BackupPlannerProblem)
    while true
        q = sample_domain(problem)

        # check if q is unsafe
        if !World.is_unsafe(problem.unsafe_zones, q)
           return q
        end
    end
end

function RRTStar.nearest(P::BackupPlannerProblem, nodes, x_rand)
    best_dist = Inf
    best_ind = -1

    for i=length(nodes):-1:1
        if (nodes[i].state[1] - x_rand[1])^2 + (nodes[i].state[2] - x_rand[2])^2 > best_dist^2
            continue
        end
        d = dubins_distance(nodes[i].state, x_rand, P.turning_radius)
        if d < best_dist
            best_dist = d
            best_ind = i
        end
    end
    return best_ind
end

function RRTStar.near(P::BackupPlannerProblem, nodes, x_new)
    ind = Int[]#deepcopy(P.root_nodes)
    # sizehint!(ind, length(nodes)/2)
    for i=length(nodes):-1:1
        if (nodes[i].state[1] - x_new[1])^2 + (nodes[i].state[2] - x_new[2])^2 > P.near_radius^2
            continue
        end
        # d = dubins_distance(nodes[i].state, x_new, P.turning_radius)
        # println(d)
        # if d < P.near_radius
            push!(ind, i)
        # end
    end
    return ind
end

function RRTStar.steer(problem::BackupPlannerProblem, x_nearest, x_rand; max_travel_dist = 11.0)

    # first plan the full dubins path
    errcode, path = Dubins.dubins_shortest_path(x_nearest, x_rand, problem.turning_radius, 1e-3)
    @assert errcode == Dubins.EDUBOK

    L = Dubins.dubins_path_length(path) 

    # get the node 20% of the waythrough
    s = min(L, max_travel_dist)

    errcode, x_new = Dubins.dubins_path_sample(path, s)
    @assert errcode == Dubins.EDUBOK
    
    return SVector{3}(x_new)
    
end

function RRTStar.collision_free(problem::BackupPlannerProblem, x_nearest, x_new; step_size=0.1)

    # create the path
    errcode, path = Dubins.dubins_shortest_path(x_nearest, x_new, problem.turning_radius, 1e-3)
    @assert errcode == Dubins.EDUBOK

    L = Dubins.dubins_path_length(path)

    x = 0.0
    while x < L
        errcode, q = Dubins.dubins_path_sample(path, x)

        @assert errcode == Dubins.EDUBOK errcode
    
        # check each point for collision 
        if q[1] < problem.domain[1][1] || q[1] > problem.domain[2][1] || q[2] < problem.domain[1][2] || q[2] > problem.domain[2][2] || World.is_unsafe(problem.unsafe_zones, q)
            return false
        end
        x += step_size
    end

    return true
end

function RRTStar.path_cost(problem::BackupPlannerProblem, x_near, x_new)
    # create the path
    x_end = @SVector [x_near[1], x_near[2], angle_add(x_near[3], pi)]
    x_start = @SVector [x_new[1], x_new[2], angle_add(x_new[3], pi)]
    errcode, path = Dubins.dubins_shortest_path(x_start, x_end, problem.turning_radius, 1e-3)
    @assert errcode == Dubins.EDUBOK    

    L = Dubins.dubins_path_length(path)

    cost = 0.0
    step_size = problem.max_velocity / problem.vo_params.updateRate
    s = 0.0
    start_step = x_start

    while s < L && isfinite(cost)
        s += step_size
        if s > L
            s = L
        end
        errcode, end_step = Dubins.dubins_path_sample(path, s)
        landmark_in_fov = false
        for l in problem.landmarks
            if is_in_fov(end_step, l, problem.vo_params.fovRadius, problem.vo_params.fovAngle, problem.vo_params.maxError) || is_in_fov(end_step, l, problem.turning_radius, float(pi), 0.0) # if in the orbit around a landmark
                landmark_in_fov = true
                break
            end
        end
         
        @assert errcode == Dubins.EDUBOK errcode
        if !landmark_in_fov
            inc_cost,_ = VO.odometry_error(start_step, end_step, problem.mapped_features[1], problem.vo_params, problem.vo_params.maxError)
            cost += inc_cost
        end
        start_step = end_step

    end
    return cost
end