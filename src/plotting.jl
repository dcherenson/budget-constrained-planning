using .RRTStar
using Plots
using Dubins
using StaticArrays

function Plots.plot!(path::DubinsPath, N = 50; kwargs...)
    errcode, samples = Dubins.dubins_path_sample_many(path, Dubins.dubins_path_length(path)/N )
    @assert errcode == Dubins.EDUBOK

    # push the final state too 
    _, pt = Dubins.dubins_path_endpoint(path)
    push!(samples, pt)

    plot!([s[1] for s in samples], [s[2] for s in samples]; kwargs...)
end

function Plots.plot!(nodes::Vector{RRTStar.Node{V}}, destination_indices::Vector{Int64}, r::F; colors = [:gray]) where {F,V} #::Vector{RRTStar.Node{SVector{3,Float64}}}
    px = [n.state[1] for n in nodes]
    py = [n.state[2] for n in nodes]
    # scatter!(px, py, label=false, opacity=0.3)

    for i in 1:length(nodes)
        node = nodes[i]
        if (node.parent_index != 0) 
            dest = div(destination_indices[i]-1,4)+1
            # Clamp dest to valid color indices
            dest = min(dest, length(colors))

            q0 = nodes[node.parent_index].state
            q1 = node.state
        
            err, path = dubins_shortest_path(q0, q1, r, 1e-3)
          
            if isempty(destination_indices)
                color = colors[1]
            else
                color = colors[dest]
            end
            plot!(path; linewidth = 3,color=color, label=false, opacity=0.3)
        end
    end
end

function plot_domain(domain::Tuple{SVector{3,F}, SVector{3,F}}) where {F}
    x0 = domain[1][1]
    y0 = domain[1][2]
    x1 = domain[2][1]
    y1 = domain[2][2]
    plot!([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]; color=:black, label=false)
end

function plotFOV(x, yaw, radius, angle; color = :yellow)

    x1 = x[1] + radius*cos(yaw - angle/2)
    y1 = x[2] + radius*sin(yaw - angle/2)
    x2 = x[1] + radius*cos(yaw + angle/2)
    y2 = x[2] + radius*sin(yaw + angle/2)
    
    # Create points for the arc
    arc_x = [x[1] + radius*cos(yaw - angle/2 + t*angle/100) for t in 0:100]
    arc_y = [x[2] + radius*sin(yaw - angle/2 + t*angle/100) for t in 0:100]
    # plot the field of view with a transparent blue fill
    fill_x = [x[1]; arc_x; x[1]]
    fill_y = [x[2]; arc_y; x[2]]
    plot!(fill_x, fill_y, fill = (0, color, 0.1), label = "", color = color)
    plot!([x[1], x1], [x[2], y1], label = "", color = :black)
    plot!([x[1], x2], [x[2], y2], label = "", color = :black)
    plot!(arc_x, arc_y, label = "", color = :black)
end

function Plots.plot!(traj::CompositeTrajectory; kwargs...)
    for p in traj.nominal_trajectory
        plot!(p; color = :red, kwargs...)
    end
    for p in traj.backup_trajectory
        plot!(p; color = :green, kwargs...)
    end
end

function plotFeatures2D(features::BallTree, color=:red)
    x   = [(features.data[i][1]) for i in 1:length(features.data)]
    y   = [(features.data[i][2]) for i in 1:length(features.data)]
    plot!(x, y, seriestype = :scatter, label = "Features", legend = :topleft, markercolor = color, msc = :auto, markersize = 2)

end

function plotFeatures2D(features::Vector{TF}, color=:red) where {TF}
    x   = [(features[i][1]) for i in 1:length(features)]
    y   = [(features[i][2]) for i in 1:length(features)]
    plot!(x, y, seriestype = :scatter, label = "Features", legend = :topleft, markercolor = color, msc = :auto, markersize = 2)

end