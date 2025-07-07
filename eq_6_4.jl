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
    using Plots
    using Glob
    using DataStructures
end

using Pkg
Pkg.add("DataStructures")

# ====================================================================================
# ====================================================================================
# loading in necessary files
# this is loading in almost 4,000 seperate shower simulations passed through detectors
@time begin
    # ===================================
    # load the reader + the file of showers 
    proj_dir = (@__DIR__) * "/tambo_sim_reader/"
    include( proj_dir * "src/read.jl" )

    # ===================================
    # first, load the full detector array
    fname =  proj_dir * "example_data/" * "modules.jld2"
    detector_full_array = read_detector_array_from_jld2(fname)

    # ================================================================================
    # create a string where we now have a specific attribute we want each file to have
    file_path = glob("retest_small_larger_valley_event_dicts_00000_000??.jld2", proj_dir * "example_data/")

    # =========================================================
    # filtering out 00000_00000 because the file does not exist
    filtered_paths = filter(path -> parse(Int, splitext(basename(path))[1][end-1:end]) ≥ 1, file_path)

    # =============================================================================================================
    # we assign each file location to our all_showers_list which now contains all the shower data from example_data
    all_showers_list = [read_triggered_showers_from_jld2(path) for path in filtered_paths]
end

# ======================================================================================
# ======================================================================================
# defining a structure to contain my position and time data from the simualted detectors
struct Detector
    # below will store the x, y, and z position of my detector
    position::NTuple{3, Float64} # this defines the field that holds the detector's spatial coordinates
    time::Float64 # field that represents when the detector was hit
end

# speed of light
c = 299_792_458.0 # defined to be in microseconds

# =====================================================================================
# let's pick the first shower.
# this object has two properties: shower.hits and shower.inj
shower = all_showers_list[2][2]

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
hits = add_detector_pos_to_hits_table(hits, detector_full_array)

# defining the lateral distances
# lets call the lateral distance the distance of the detector from the shower core for now
detector_simulation = DataFrame(hits)

# ============================================================
# ============================================================
# here is where we will define the core and detector positions
# setting up our first hits
first_hits = assign_first_hit_detectors(detector_simulation)
detector_hits = setup_first_hit_detectors(first_hits)

# ================================================================================================
# lets walk through the process of finding the timing log-likelihood
# lets begin by getting an estimated direction of where the shower is coming from/heading whatever
begin
    result = Minuit(chi_sq, 0.0, 0.0, 1e-5, names=("theta", "phi", "t0"))
    println(result)
    result.limits["phi"] = [0.0, 2*pi]
    result.limits["theta"] = [0.0, pi]
    new = migrad!(result)
    guess_dir = get_dir_vec(result.values["theta"], result.values["phi"])
    estimate = -guess_dir
    core_time_estimation = result.values["t0"]
end
# ===================================================================================================
# okay, now that we have a direction for where the shower is heading, lets see how this can be useful
# let's go ahead and rotate the detector array to be flat, and the parabaloid function
# or eq_5_11 to be in the direction of the estimated directional vector
# ===========================================================================================
# determining our rotation matrix
# observation plane vector is defined as zcorsika, we use this to create an xyzcorsika matrix
# defining our x, y, and z corsika vectors
begin
    zcorsika = SVector{3}(estimate)
    xcorsika = SVector{3}([0,-zcorsika[3]/sqrt(zcorsika[2]^2+zcorsika[3]^2),zcorsika[2]/sqrt(zcorsika[2]^2+zcorsika[3]^2)])
    ycorsika = cross(zcorsika,xcorsika)

    # xyz corsika matrix is defined here
    xyzcorsika = [

        xcorsika.x xcorsika.y xcorsika.z;
        ycorsika.x ycorsika.y ycorsika.z;
        zcorsika.x zcorsika.y zcorsika.z;
    ]

    xyzcorsika_inverse = inv(xyzcorsika)
end
# =====================================================================================================================
# lets use our designated shower to find a plane fit equation describing the time delay with respect to the plane front

# here is where we will rotate our detector array to be flat relative to the gaussian shower represented by eq 5_11
rotated_detector_hits = [xyzcorsika_inverse * SVector{3}(detector_hits[i].position) for i in eachindex(detector_hits)]

# here is where we will rotate the plane equation to be in the same direction as the estimated vector
r = [
    sqrt(((rotated_detector_hits[i][1])^2 + (rotated_detector_hits[i][2])^2 + (rotated_detector_hits[i][3])^2)) 
    for i in eachindex(detector_hits)
]

"""
    delta_t(a, b)

    we're finding the time delay with respect to the plane front by plugging in the lateral distance of detector positions relative to the core.
"""
function delta_t(a, b)

    function five_elev_calc(lateral)
            
            # defining equation 5.11
            δ = a*lateral^2 + b
    end
    return sum(five_elev_calc.(r))
end

result = Minuit(delta_t, 0.0, 0.0, names=("a", "b"))
migrad!(result)
opt_params = (result.values["a"], result.values["b"])

# =================================================================
# we need to find our d values now

d = [((opt_params[1]*r[i]^2 + opt_params[2]) - ((estimate[1]*detector_hits[i].position[1] + estimate[2]*detector_hits[i].position[2])/estimate[3])) for i in eachindex(detector_hits)]

front_time = d ./ c

# ===================================================================================
# lets make equation 6_4
"""
    function timing_likely()
    
    this function will take in () and spit back out the likelihood of an 
"""

function timing_likely(front_times)

    
end

# ===================================================================================
# dont erase anything here because just in case i need it, i have the thought process
begin
    # =================================================================================
    # lets calculate lateral distance
    # to find all the info we need on position, we need the following attributes:
    # direction vector, initial and terminal point, distance traveled
    # without any two of these, we can't get the info we need to make calculates
    # ==========================================================================
    # here is where we will find the distance a `hit` traveled using speed, time, dist.

    # =================================================================================
    # here is where we will find the coordinate of the particle in the shower thickness

    # ===============================================================================================
    # here is where we will find the vector from that coordinate we found before and the core position

    # ========================================================================================================
    # here is where we will `swing` that vector until we find both the angle and the arc length of that vector
end