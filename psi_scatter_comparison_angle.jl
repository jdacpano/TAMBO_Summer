# ==========================
# loading in necessary files
begin
    # load the reader + the file of showers 

    proj_dir = (@__DIR__) * "/tambo_sim_reader/"
    include( proj_dir * "src/read.jl" )

    # ===================================
    # first, load the full detector array
    fname =  proj_dir * "example_data/" * "modules.jld2"
    detector_array = read_detector_array_from_jld2( fname )

    # ====================================
    # this is the.jld2 file of our showers
    fname = proj_dir * "example_data/" * "retest_small_larger_valley_event_dicts_00000_00002.jld2"

    # lets load every shower from the .jld2 file.
    # this produces a huge table of showers...
    all_showers = read_triggered_showers_from_jld2( fname )
end
# ======================================
# loading in the packages we'll be using
begin
    using Optim
    using Minuit2
    using Parquet2
    using DataFrames
    using LinearAlgebra
    using YAML
    using StaticArrays
    using Statistics
    using StatsBase
    using Makie
    using Plots
    using CairoMakie
    using Unitful
    using MeshGrid
    using LaTeXStrings
    using MathTeXEngine: FontFamily, texfont
end

# ============================
# let's pick the first shower.
# this object has two properties: shower.hits and shower.inj
shower = all_showers[10]

# shower.inj is a "NamedTuple" containing the true properties that were "injected" into the simulation in order to generate the shower. these properties include:
# - the energy + direction of the initial tau neutrino
# - the energy + direction of the tau produced when the tau neutrino interacted.
inj_truth = shower.inj

# shower.hits is a table containing all the particles that hit the detector. 
# each particle has the properties: 
# - `detector` = the number of the detector it hit,
# - `pdg` = the type of particle (22 = photon),
# - `kinetic_energy` = the energy of the particle,
# - `time` = the time at which the particle hit the detector.
hits = shower.hits

# instead of just knowing the numbers of each detector, it would be nice to also have the locations of each detector easily accessible. let's add that information to `hits:`
hits = add_detector_pos_to_hits_table(hits, detector_array)

# ======================================================================================
# defining a structure to contain my position and time data from the simualted detectors
struct Detector
    # below will store the x, y, and z position of my detector
    position::NTuple{3, Float64} # this defines the field that holds the detector's spatial coordinates
    time::Float64 # field that represents when the detector was hit
end

# speed of light
c = 299_792_458.0 # defined to be in microseconds

# ===================================================================================
# write a function which creates a dataframe using the first 10 showers detector data
# by using the angular difference and primary energy of each shower

"""
function psi_graph_scatter

    this function will take in shower data and return a datframe containing the estimated direction vector, true direction vector, angular difference between the true and estimated, and the primary energy of each shower `run`

"""
function psi_graph_scatter(shower)

    # we need to initialize an empty vector to contain our data
    prim_energy = Float64[]
    angular_difference = Float64[]
    estimated_direction = Vector{Vector{Float64}}()
    true_direction = Vector{Vector{Float64}}()

    for i in eachindex(all_showers)
        shower = all_showers[i]
        inj_truth = shower.inj
        hits = shower.hits
        hits = add_detector_pos_to_hits_table(hits, detector_array)

        # we want to first define our shower data to an accessible dataframe
        # assigning hits table to a dataframe which for me is easier to use and read
        detector_simulation = DataFrame(hits)
            
        # this is where we redefine the hits dataframe to only use first hit detector times
        first_hits = assign_first_hit_detectors(detector_simulation)

        # setup our detector structure with the correct spatial coordinates(x,y,z) adn hit times
        detector_hits = setup_detectors(first_hits)
            
        function chi_sq(theta, phi, t0)

            nx, ny, nz = get_dir_vec(theta, phi)
                
            function calc(detector)
                    
                squared = 
                    (
                        (detector.time - t0 
                        + (nx * detector.position[1]/ c) 
                        + (ny * detector.position[2]/ c)
                        + (nz * detector.position[3]/ c))^2
                    )
                end
            return sum(calc.(detector_hits))
        end

        # n is already normalized btw (no need to normalize again)
        n, core_arrival_time = optimizer_plane_fit(chi_sq)

        xdir = sin(inj_truth.tau_θ) * cos(inj_truth.tau_ϕ)
        ydir = sin(inj_truth.tau_θ) * sin(inj_truth.tau_ϕ)

        true_directional_shower = [xdir, ydir]
        guess_dir = [-n[1], -n[2]]

        true_directional_shower_norm = true_directional_shower ./ norm(true_directional_shower)
        guess_dir_norm = Vector{Float64}([-n[1], -n[2]] ./ norm(guess_dir))

        core_θ_guess_true = calc_great_circle_distance(true_directional_shower_norm, guess_dir_norm)
        θ = rad2deg(core_θ_guess_true)

        # pushing our results to the parallel arrays
        push!(prim_energy, round(inj_truth.tau_energy / 1PeV,digits=2))
        push!(angular_difference, round(θ, digits=2))
        push!(estimated_direction, guess_dir_norm)
        push!(true_direction, true_directional_shower_norm)
    end
    
    # creating the dataframe where we will use this data in order to graph our scatter plot
        psi_df = DataFrame(
            "energy" => prim_energy,
            "θ_difference" => angular_difference, 
            "estim_dir" => estimated_direction,
            "true_dir" => true_direction    
        )
    return psi_df
end

psi_df = psi_graph_scatter(shower)

sampled_xs = psi_df.energy
sampled_ys = psi_df.θ_difference

f = Figure() 
ax = Axis(f[1, 1], 
title = "Angular Difference",
ylabel = L"\psi = - \theta_\mathrm{true} - \theta_\mathrm{estimated}",
xlabel = "Energy in PeV",
aspect = 1, 
)

Makie.scatter!(ax, sampled_xs, sampled_ys; markersize = 6)

save("/Users/carlitos/Desktop/TAMBO/traj_pics/psi_angle_true_compare_6_30.png", f)

display(f)