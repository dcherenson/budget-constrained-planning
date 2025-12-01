"""
Animation plotting functions for budget-constrained planning simulations.
"""

using Plots
using Images

"""
    find_destination_index(nodes, start)

Helper function to traverse backup planner tree to find destination node.
Used for visualization of tree paths.
"""
function find_destination_index(nodes, start::Int64)
    if nodes[start].parent_index == 0
        return start
    else
        return find_destination_index(nodes, nodes[start].parent_index)
    end
end

"""
    compute_fov_coverage(x_hist, vo_params, domain_size, AR; resolution=1.0)

Compute which areas have been seen by the FOV up to each frame.
Returns a BitArray for each frame indicating coverage.
"""
function compute_fov_coverage(x_hist, vo_params, domain_size, AR; resolution=1.0)
    nx = Int(ceil(domain_size * AR / resolution))
    ny = Int(ceil(domain_size / resolution))
    
    coverage_hist = Vector{BitMatrix}()
    cumulative_coverage = falses(ny, nx)
    
    for i in 1:length(x_hist)
        x = x_hist[i]
        
        # Mark all points within FOV as seen
        for ix in 1:nx
            for iy in 1:ny
                if cumulative_coverage[iy, ix]
                    continue  # Already seen
                end
                
                # Grid point coordinates
                px = (ix - 0.5) * resolution
                py = (iy - 0.5) * resolution
                
                # Check if point is in FOV
                dx = px - x[1]
                dy = py - x[2]
                dist = sqrt(dx^2 + dy^2)
                
                if dist <= vo_params.fovRadius
                    # Check angle constraint
                    angle_to_point = atan(dy, dx)
                    heading = x[3]
                    angle_diff = abs(atan(sin(angle_to_point - heading), cos(angle_to_point - heading)))
                    
                    if angle_diff <= vo_params.fovAngle / 2
                        cumulative_coverage[iy, ix] = true
                    end
                end
            end
        end
        
        push!(coverage_hist, copy(cumulative_coverage))
    end
    
    return coverage_hist
end

"""
    animate_simulation(x_hist, x_est_hist, gk_hist, features_hist, backup_hist,
                       nominal_planner, backup_planner, vo_params, domain_size, AR;
                       cost_hist=nothing, num_feats_hist=nothing, landmarks_discovered_hist=nothing,
                       image_path="images/field.png", airplane_path="images/blue_airplane.png",
                       airplane_size=15.0, frame_skip=1, show_estimate=true, show_unseen=true,
                       save_frames=nothing, output_dir="frames", frame_ext="png",
                       show_text=true, show_labels=true)

Create animation of the full simulation showing:
- Background field image
- Unseen areas (transparent light orange, if show_unseen=true)
- Features
- Backup planner tree
- Nominal path
- True trajectory (blue)
- Estimated trajectory (red, if show_estimate=true)
- Field of view
- Committed trajectory
- Landmarks (only shown once discovered)
- Airplane icon

Arguments:
- cost_hist: Optional cost history to display in bottom right corner
- num_feats_hist: Optional number of features history to display overlapping features
- landmarks_discovered_hist: Optional history of which landmarks have been discovered
- frame_skip: Plot every nth frame (1 = all frames, 2 = every other frame, etc.)
- show_estimate: Whether to show estimated trajectory alongside true trajectory
- show_unseen: Whether to overlay transparent orange on unseen areas
- save_frames: Optional vector of frame numbers to save as images (e.g., [10, 50, 100])
- output_dir: Directory to save captured frames (default: "frames")
- frame_ext: File extension for saved frames (default: "png", can be "svg", "pdf", etc.)
- show_text: Whether to show text annotations (error, overlapping features)
- show_labels: Whether to show axis labels (East [m], North [m])
"""
function animate_simulation(x_hist, x_est_hist, gk_hist, features_hist, backup_hist,
                            nominal_planner, backup_planner, vo_params, domain_size, AR;
                            cost_hist=nothing,
                            num_feats_hist=nothing,
                            landmarks_discovered_hist=nothing,
                            image_path="images/field.png", 
                            airplane_path="images/blue_airplane.png",
                            airplane_size=15.0, 
                            frame_skip=1,
                            show_estimate=true,
                            show_unseen=true,
                            save_frames=nothing,
                            output_dir="frames",
                            frame_ext="png",
                            show_text=true,
                            show_labels=true)
    
    image = load(image_path)
    airplane_img = load(airplane_path)
    
    # Create output directory if saving frames
    if !isnothing(save_frames) && !isempty(save_frames)
        if !isdir(output_dir)
            mkpath(output_dir)
        end
        println("Will save frames: $save_frames to $output_dir/ as .$frame_ext files")
    end
    
    # Precompute FOV coverage for all frames if needed
    coverage_hist = nothing
    if show_unseen
        println("Computing FOV coverage...")
        coverage_hist = compute_fov_coverage(x_hist, vo_params, domain_size, AR, resolution=2.0)
    end

    frames = isnothing(save_frames) ? (1:frame_skip:length(x_hist)) : save_frames
    
    anim = @animate for i in frames
        plot()
        
        # Background image
        plot!([0, domain_size*AR], [0, domain_size], 
              reverse(image, dims=1), yflip=false, aspect_ratio=:auto)
        
        # Overlay unseen areas with transparent light orange
        if show_unseen && !isnothing(coverage_hist)
            coverage = coverage_hist[i]
            ny, nx = size(coverage)
            resolution = domain_size * AR / nx
            
            for ix in 1:nx
                for iy in 1:ny
                    if !coverage[iy, ix]
                        # This area has not been seen yet
                        x_left = (ix - 1) * resolution
                        x_right = ix * resolution
                        y_bottom = (iy - 1) * resolution
                        y_top = iy * resolution
                        
                        plot!(Shape([x_left, x_right, x_right, x_left],
                                   [y_bottom, y_bottom, y_top, y_top]),
                             color=:orange, alpha=0.3, linewidth=0)
                    end
                end
            end
        end
        
        # Features - plot all mapped features in white first
        plotFeatures2D(features_hist[i], :white)
        
        # Then plot currently visible features in red on top
        visible_idx = features_in_fov_idx(features_hist[i], 
                                          x_hist[i], vo_params.fovRadius, vo_params.fovAngle)
        if !isempty(visible_idx)
            visible_features = [features_hist[i].data[j] for j in visible_idx]
            plotFeatures2D(visible_features, :red)
        end
        
        # Backup planner tree
        destination_indices_back = zeros(Int64, length(backup_hist[i]))
        for j in 1:length(backup_hist[i])
            destination_indices_back[j] = find_destination_index(backup_hist[i], j)
        end
        plot!(backup_hist[i], destination_indices_back, backup_planner.prob.turning_radius, 
              colors=[:orange, [:white for k=1:length(backup_planner.prob.landmarks)-2]..., :yellow])
        
        # Nominal path from current position
        success, best_path = query_nominal_planner(nominal_planner, x_est_hist[i])
        if success
            best_path = make_path_from_waypoints(best_path, nominal_planner.prob.turning_radius)
            for p in best_path
                plot!(p, linewidth=5, color=:black, linestyle=:solid)
            end
        end
        
        # True trajectory
        x_hist_x = [x[1] for x in x_hist[1:i]]
        x_hist_y = [x[2] for x in x_hist[1:i]]
        plot!(x_hist_x, x_hist_y, color=:blue, linewidth=5, linestyle=:solid, label="True")
        
        # Estimated trajectory (if requested)
        if show_estimate && !isempty(x_est_hist)
            x_est_x = [x[1] for x in x_est_hist[1:i]]
            x_est_y = [x[2] for x in x_est_hist[1:i]]
            plot!(x_est_x, x_est_y, color=:red, linewidth=3, linestyle=:dash, 
                  alpha=0.7, label="Estimated")
        end
        
        # Field of view
        plotFOV(x_hist[i][1:2], x_hist[i][3], vo_params.fovRadius, vo_params.fovAngle)
        shift = vo_params.maxError / sin(vo_params.fovAngle / 2)
        x_shifted = x_hist[i][1:2] +  SVector(cos(x_hist[i][3]), sin(x_hist[i][3])) * shift 
        plotFOV(x_shifted, x_hist[i][3], vo_params.fovRadius - shift - vo_params.maxError, vo_params.fovAngle, color=:blue)
        
        # Committed trajectory
        plot!(gk_hist[i]; linewidth=5)
        
        # Determine which landmarks to show based on discovery
        if !isnothing(landmarks_discovered_hist) && i <= length(landmarks_discovered_hist)
            discovered = landmarks_discovered_hist[i]
        else
            # If no history provided, show all landmarks
            discovered = [true for _ in 1:length(backup_planner.prob.landmarks)]
        end
        
        # Goal landmark (only if discovered)
        plot!(Shape(backup_planner.prob.landmarks[end][1] .+ backup_planner.prob.turning_radius*cos.(0:0.01:2pi),
                    backup_planner.prob.landmarks[end][2] .+ backup_planner.prob.turning_radius*sin.(0:0.01:2pi)), 
                color=:cyan, alpha=0.2)
        plot!([backup_planner.prob.landmarks[end][1]], [backup_planner.prob.landmarks[end][2]], 
                color=:yellow, marker=:circle, markersize=10)
    
        # Start landmark (only if discovered)
        plot!(Shape(backup_planner.prob.landmarks[1][1] .+ backup_planner.prob.turning_radius*cos.(0:0.01:2pi),
                    backup_planner.prob.landmarks[1][2] .+ backup_planner.prob.turning_radius*sin.(0:0.01:2pi)), 
                color=:orange, alpha=0.2)
        plot!([backup_planner.prob.landmarks[1][1]], [backup_planner.prob.landmarks[1][2]], 
                color=:orange, marker=:circle, markersize=10)
        
        # Mid landmarks (only if discovered)
        for j in 2:length(backup_planner.prob.landmarks)-1
            if discovered[j]
                plot!(Shape(backup_planner.prob.landmarks[j][1] .+ backup_planner.prob.turning_radius*cos.(0:0.01:2pi),
                            backup_planner.prob.landmarks[j][2] .+ backup_planner.prob.turning_radius*sin.(0:0.01:2pi)), 
                      color=:white, alpha=0.2)
                plot!([backup_planner.prob.landmarks[j][1]], [backup_planner.prob.landmarks[j][2]], 
                      color=:white, marker=:circle, markersize=10)
            end
        end
        
        # Domain boundaries
        plot_domain(nominal_planner.prob.domain)
        plot!(nominal_planner.prob.unsafe_zones; color=:red, width=2)
        
        # Estimated airplane position (transparent red) - plot first so true position is on top
        if show_estimate && !isempty(x_est_hist) && i <= length(x_est_hist)
            rotated_airplane_est = imrotate(airplane_img, -x_est_hist[i][3], axes(airplane_img))
            h_plane_est, w_plane_est = size(rotated_airplane_est)
            aspect_est = w_plane_est / h_plane_est
            half_size_est = airplane_size / 2
            plot!([x_est_hist[i][1] - half_size_est*aspect_est, x_est_hist[i][1] + half_size_est*aspect_est],
                  [x_est_hist[i][2] - half_size_est, x_est_hist[i][2] + half_size_est],
                  reverse(rotated_airplane_est, dims=1), yflip=false, alpha=0.3)
        end
        
        # True airplane position
        rotated_airplane = imrotate(airplane_img, -x_hist[i][3], axes(airplane_img))
        h_plane, w_plane = size(rotated_airplane)
        aspect = w_plane / h_plane
        half_size = airplane_size / 2
        plot!([x_hist[i][1] - half_size*aspect, x_hist[i][1] + half_size*aspect],
              [x_hist[i][2] - half_size, x_hist[i][2] + half_size],
              reverse(rotated_airplane, dims=1), yflip=false)
        
        # Compute overlapping features - use num_feats_hist if available
        if !isnothing(num_feats_hist) && i <= length(num_feats_hist)
            num_feats = isnan(num_feats_hist[i]) ? 0 : Int(num_feats_hist[i])
        else
            # Fallback: compute overlapping features between current and previous frame
            num_feats = 0
            if i > 1
                prev_visible_idx = features_in_fov_idx(features_hist[i-1], 
                                                       x_hist[i-1], vo_params.fovRadius, vo_params.fovAngle)
                curr_visible_idx = features_in_fov_idx(features_hist[i], 
                                                       x_hist[i], vo_params.fovRadius, vo_params.fovAngle)
                num_feats = length(intersect(Set(prev_visible_idx), Set(curr_visible_idx)))
            end
        end
        
        # Add text in bottom right corner
        if show_text
            text_x = domain_size * AR * 0.98
            text_y_base = domain_size * 0.05
            if !isnothing(cost_hist) && i <= length(cost_hist)
                annotate!(text_x, text_y_base + 8, 
                         text("Error: $(round(cost_hist[i], digits=2)) m", 10, :white, :right))
            end
            annotate!(text_x, text_y_base, 
                     text("Overlapping Features: $num_feats", 10, :white, :right))
        end
        
        # Formatting
        plot!(legend=false, axis=false, grid=false, widen=false,
              background_color=:transparent, foreground_color=:black,
              margin=0Plots.mm, left_margin=0Plots.mm, right_margin=0Plots.mm,
              top_margin=0Plots.mm, bottom_margin=0Plots.mm)
        xlims!(0, domain_size*AR)
        ylims!(0, domain_size)
        if show_labels
            xlabel!("East [m]")
            ylabel!("North [m]")
        end
        
        # Save frame if requested
        if !isnothing(save_frames) && i in save_frames
            frame_filename = joinpath(output_dir, "frame_$(lpad(i, 4, '0')).$frame_ext")
            savefig(frame_filename)
            println("Saved frame $i to $frame_filename")
        end
    end
    
    return anim
end

"""
    animate_orbit(x0, nominal_planner, backup_planner, domain_size, AR;
                  image_path="images/field.png", airplane_path="images/blue_airplane.png",
                  airplane_size=15.0, dt=0.1)

Create animation of vehicle orbiting around start landmark.
"""
function animate_orbit(x0, nominal_planner, backup_planner, domain_size, AR;
                      image_path="images/field.png",
                      airplane_path="images/blue_airplane.png",
                      airplane_size=15.0,
                      dt=0.1)
    
    image = load(image_path)
    airplane_img = load(airplane_path)
    
    start_yaw = x0[3]
    t_orbit_end = 2*π*nominal_planner.prob.turning_radius / nominal_planner.prob.max_velocity
    num_frames = Int(ceil(t_orbit_end / dt))
    
    anim = @animate for i in 1:num_frames
        t_orbit = (i-1)*dt
        plot()
        
        # Background
        plot!([0, domain_size*AR], [0, domain_size],
              reverse(image, dims=1), yflip=false, aspect_ratio=:auto)
        
        # Goal landmark
        plot!(Shape(backup_planner.prob.landmarks[end][1] .+ backup_planner.prob.turning_radius*cos.(0:0.01:2pi),
                    backup_planner.prob.landmarks[end][2] .+ backup_planner.prob.turning_radius*sin.(0:0.01:2pi)), 
              color=:cyan, alpha=0.2)
        plot!([backup_planner.prob.landmarks[end][1]], [backup_planner.prob.landmarks[end][2]], 
              color=:yellow, marker=:circle, markersize=10)
        
        # Start landmark (orbit center)
        plot!(Shape(backup_planner.prob.landmarks[1][1] .+ backup_planner.prob.turning_radius*cos.(0:0.01:2pi),
                    backup_planner.prob.landmarks[1][2] .+ backup_planner.prob.turning_radius*sin.(0:0.01:2pi)), 
              color=:orange, alpha=0.2)
        plot!([backup_planner.prob.landmarks[1][1]], [backup_planner.prob.landmarks[1][2]], 
              color=:orange, marker=:circle, markersize=10)
        
        # Domain
        plot_domain(nominal_planner.prob.domain)
        plot!(nominal_planner.prob.unsafe_zones; color=:red, width=2)
        
        # Compute orbit position
        yaw = mod(start_yaw + t_orbit * nominal_planner.prob.max_velocity / nominal_planner.prob.turning_radius, 2π)
        center = backup_planner.prob.landmarks[1]
        x_pos_orbit = SVector(
            center[1] + nominal_planner.prob.turning_radius * cos(yaw - π/2),
            center[2] + nominal_planner.prob.turning_radius * sin(yaw - π/2),
            yaw
        )
        
        # Airplane at orbit position
        rotated_airplane = imrotate(airplane_img, -x_pos_orbit[3], axes(airplane_img))
        h_plane, w_plane = size(rotated_airplane)
        aspect = w_plane / h_plane
        half_size = airplane_size / 2
        plot!([x_pos_orbit[1] - half_size*aspect, x_pos_orbit[1] + half_size*aspect],
              [x_pos_orbit[2] - half_size, x_pos_orbit[2] + half_size],
              reverse(rotated_airplane, dims=1), yflip=false)
        
        # Formatting
        plot!(legend=false, axis=false, grid=false, widen=false,
              background_color=:transparent, foreground_color=:black)
        xlims!(0, domain_size*AR)
        ylims!(0, domain_size)
    end
    
    return anim
end

"""
    save_animation(anim, filename; fps=10)

Save animation to file.
"""
function save_animation(anim, filename; fps=10)
    gif(anim, filename, fps=fps)
end
