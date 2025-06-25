
using Optim
using Parquet2
using DataFrames
using LinearAlgebra
using YAML
using StaticArrays
using Statistics
using StatsBase
using Makie
using Plots
# for plotting
using CairoMakie

# ============================================================
# run them through my optimizer. + get a best guess direction.
# speed of light
    c = 299_792_458.0/1e6 # needs to be in microseconds

    # defining a structure to contain my position and time data
        struct Detector
            # below will store the x, y, and z position of my detector
            position::NTuple{2, Float64} # this defines the field that holds the detector's spatial coordinates
            time::Float64 # in microseconds, a field that represents when the detector was hit
        end
# =======================================
# load in the hits from the parquet file.
    pqf = Parquet2.Dataset("/Users/carlitos/Desktop/TAMBO/trajectory")
    df = DataFrame(pqf)
# =====================================
# create a detector array from the hits `normalized` with respect to the core position of the detectors
    # t = df.time
    # median_value = median(t)
    
    # df[738, :]

    # df.time[738]

    # median_value = argmed(t)
    """
    get_core_position(arr)

    will take the input of any array and output the median index of that array.
    """
    function get_core_position(arr)
        m  = median(arr)
        _, idx = findmin(abs.(arr .- m))
        return idx
    end

    """
    setup_detectors(dataframe)

    this function will take in a parquet file of hits `dataframe` and will pass us back a structure of detectors which contain the shower coordinates of the hits: x, y, and t. these hits are `normalized` with respect to the core arrival index.
    """
    function setup_detectors(dataframe)

        # extract the core position
            t = df.time
            median_value = argmed(t)
        # subtract off the core position from each hit position and created a detector structure
            detector_array = [Detector((df.x[i] .- df.x[median_value], df.y[i] .- df.y[median_value]), t[i] .- t[median_value]) for i in eachindex(df.x)]

            return detector_array
    end

    particle_detectors = setup_detectors(df)
    # particle_detectors[1, :]
    
    # particle_detectors = [Detector((df.x[i], df.y[i]), df.time[i]) for i in eachindex(df.x)]

    # # assigning x and y values from parquet file to an accessible array
    #     xs = df.x
    #     ys = df.y
    #     zs = df.z
    #     t = df.time # vector of time values
    
    # alternate location where we subtract first hit position
    # begin
    #         first_hit_index = argmin(t)
    #         println(df[149, :])

    #         particle_detectors = [Detector((df.x[i] .- df.x[first_hit_index], df.y[i] .- df.y[first_hit_index]), df.time[i] .- df.time[first_hit_index]) for i in eachindex(df.x)]

    #         # assigning x and y values from parquet file to an accessible array
    #             xs = [normalize_particle_detectors[i].position[1] for i in eachindex(df.x)]
    #             ys = [normalize_particle_detectors[i].position[2] for i in eachindex(df.x)]
    #             t = [normalize_particle_detectors[i].time for i in eachindex(df.x)]
    # end
# ================================
# defining my chi_squared function

    """
    function chi_sq(params)

    Calculates the chi-sq (Eq. 6.2 in ...) given the time, x-y positions of hits on a detector.
    These are passed in as a vector of detector structs.

    The chi-sq is a number which indicates how close we are to the true angle. Closer approximations to the true angle results in a smaller chi-sq number, and the counter for if the true angle is far from our guess.
    """
    function chi_sq(params)

        # change to make: add another parameter for `t0` = the time at which the shower core hits the surface. 
        nx, ny, t0 = params
        # nx, ny = (nx, ny)./sqrt(nx^2+ny^2)

        function calc(detector)
            squared = ((detector.time - t0 + (nx * detector.position[1]/ c) + (ny * detector.position[2]/ c))^2)
        end

        return sum(calc.(particle_detectors))
    end
    
    begin
        chi_sq([0.1, 0.1, 0.0])

        result = optimize(chi_sq, [0.1, 0.1, 0.0])
        opt_params = Optim.minimizer(result)
        
        # getting back our chi_squared optimized values
        core_time_estimation = opt_params[3]
        nx, ny = opt_params[1], opt_params[2]
        
        # normalizing our chi_squared `guess` vector
        nx, ny = (nx, ny) ./ sqrt(opt_params[1]^2 + opt_params[2]^2)

        guess_dir = [nx, ny, 0.0]
    end
# ===========================
# load in the true direction
begin
    data1 = YAML.load_file("config(12).yaml")
    println(data1)

    peru_true_zenith = deg2rad(data1["zenith"][1])
    peru_true_azimuth = deg2rad(data1["azimuth"][1])

    peru_true_vector = [
        cos(peru_true_azimuth)*sin(peru_true_zenith), # x
        sin(peru_true_azimuth)*sin(peru_true_zenith), # y
        cos(peru_true_zenith) # z
    ]
end
# =================================================
begin
    # load in the rotation matrix + check that it works
        data2 = YAML.load_file("config-2.yaml")
        println(data2)

        zcorsika = SVector{3}(data2["plane"]["normal"])
        xcorsika = SVector{3}([0,-zcorsika[3]/sqrt(zcorsika[2]^2+zcorsika[3]^2),zcorsika[2]/sqrt(zcorsika[2]^2+zcorsika[3]^2)])
        ycorsika = cross(zcorsika,xcorsika)

    # xyz corsika matrix is defined here
        xyzcorsika = [

            xcorsika.x xcorsika.y xcorsika.z;
            ycorsika.x ycorsika.y ycorsika.z;
            zcorsika.x zcorsika.y zcorsika.z;
            ]

        xyzcorsika_inverse = inv(xyzcorsika)

        true1 = xyzcorsika * xcorsika
        true2 = xyzcorsika * ycorsika
        true3 = xyzcorsika * zcorsika
end
# ==========================================================
# rotate the true direction from Peru coords. to flat coords

    true_directional_shower = normalize(xyzcorsika * peru_true_vector)

# ================================================================================
# compare the algorithims guess direction to the true direction of the air shower!

    println(true_directional_shower)
    println(guess_dir)

    # angle calculation between guess direction and true direcrtion of the air shower
    # great circle distance calculation (simple dot product, acos calculation)


    """
    function calc_great_circle_distance(dir1, dir2)

    this function will take two directional vectors and find the angle between the two vectors.
    """
    function calc_great_circle_distance(dir1, dir2)

        θ_guess_true = acos(dot(dir1, dir2)/norm(dir1) * norm(dir2))

        return θ_guess_true
    end

    core_θ_guess_true = calc_great_circle_distance(true_directional_shower, guess_dir)
    rad2deg(core_θ_guess_true)

    # getting the angle of each vector from the x-axis
    # θ_directional_shower = atan(directional_shower[2], directional_shower[2])
    # θ_guess_dir = atan(guess_dir[2], guess_dir[1])

    # θ_start, θ_end = sort([θ_directional_shower, θ_guess_dir]) # Ensure correct sweep

    # Plot arc for angle
    # r = 0.2 * min(directional_shower, guess_dir) # arc radius
    # r = 0.2 * min(norm(directional_shower), norm(guess_dir)) # arc radius
    # numb_points = 100
    # arc_x = [r * cos(t) for t in range(θ_start, θ_end, length=numb_points)]
    # arc_y = [r * sin(t) for t in range(θ_start, θ_end, length=numb_points)]

    # Optional label for angle
    # mid_angle = (θ_start + θ_end) / 2
    # annotate!(0.9 * cos(mid_angle), 0.9 * sin(mid_angle), text("θ = $(round(rad2deg(θ), digits=2))°", :black))
    
# ===========================================
# make a plot, which should have three things
# we should have our true directional_shower vector, guess_dir, particle points
# ==============
# plot ...

begin

        fig = Figure(size=(500, 300))
        ax = Axis(fig[1, 1], xlabel = "x[m]", ylabel = "y[m]")
        colsize!(fig.layout, 1, Aspect(1,1.0))
        
        # plots all of the particles
        # scatter!(ax, xs, ys; color=:blue, markersize = 6)

        # this plots our time evolution of the particles in parquet file
        low, high = quantile(t, [0.25, 0.75]) # stretch over the 1st to 99th percentile

        scatter!(ax, xs, ys;
        
            # plot in linear scale
            color = t,
            colorrange=(low, high),

            # # plot in log scale
            # color=log10.(t),
            colormap=:blues,
            markersize = 6
        )

        Colorbar(fig[1,2], 
            limits=(low, high), 
            colormap=:blues, label="Time"
        )

        # plots our directional_shower vector and guess_dir
        # arrows = arrows2d!(ax, 
        #     [Point2f(-1000, 2400), Point2f(-1000, 2400)], 
        #     [directional_shower[[1,2]], guess_dir[[1,2]]]; 
        #     #
        #     lengthscale=1000, color=[:red, :skyblue])

        # plot my arrows and keep their handles
        arrow1 = arrows2d!(ax,
            [Point2f(0, 0)],
            [true_directional_shower[[1,2]]];
            lengthscale=1000, color=:red)

        arrow2 = arrows2d!(ax,
            [Point2f(0, 0)],
            [guess_dir[[1,2]]];
            lengthscale=1000, color=:skyblue)

        # angles = LinRange(θ_start, θ_end, 40)
        # radii = LinRange(0, 2, 40)
        # polar(angles, radii, title="Polar Plot of Angles and Radii")

        Label( fig[1,1,Top()], "Core Center Angular Difference = $(rad2deg(core_θ_guess_true))°" )

        # create my legend
        legend = Legend(fig[1, 1],
        [
            MarkerElement(color = :red, marker = :utriangle),
            MarkerElement(color = :skyblue, marker = :utriangle)
        ],
        ["True Directional Vector", "Reconstructed Directional Vector"];
        labelsize = 10,
        patchsize = (10, 10),
        tellheight = false,
        halign = :left,
        valign = :bottom,
        )

    save("/Users/carlitos/Desktop/TAMBO/traj_pics/core_guess_true6_25_25.png", fig)

    fig
end