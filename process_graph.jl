using Images, FileIO, RealNeuralNetworks
using RealNeuralNetworks.Manifests
using PyCall

# Import the memory_profiler module from Python
memory_profiler = pyimport("memory_profiler")

# Define the @profile macro to use the memory_profiler decorator
macro profile(expr)
    quote
        memory_profiler.profile($(esc(expr)))
    end
end

println("it starts")

# Define the directory path and image path
dir_path = "/burg/home/jd4068/RNN/Morphology/NET10_DownSamp"
img_path = joinpath(dir_path, "NET10_s0216.png")

# Save memory usage data to a file (1st checkpoint)
open("memory_usage_checkpoint1.dat", "w") do io
    redirect_stdout(io) do
        memory_profiler.memory_usage((load_images,), interval=0.1, timeout=3600)
    end
end


# Load the first image to determine size
img = load(img_path)

# Save memory usage data to a file (2st checkpoint)
open("memory_usage_checkpoint2.dat", "w") do io
    redirect_stdout(io) do
        memory_profiler.memory_usage((load_images,), interval=0.1, timeout=3600)
    end
end


# Read all image file paths in the directory
pathes = readdir(dir_path)

# Log the number of images and size of the first image
println("Number of images: ", length(pathes))
println("Size of the first image: ", size(img))

# Preallocate volume array
vol_size = (size(img)..., length(pathes))
vol = zeros(Float64, vol_size)  # Using Float64 for compatibility; adjust as needed

# Save memory usage data to a file (3st checkpoint)
open("memory_usage_checkpoint3.dat", "w") do io
    redirect_stdout(io) do
        memory_profiler.memory_usage((load_images,), interval=0.1, timeout=3600)
    end
end


# Load and stack each image into the volume
@profile function load_images()
    for z in 1:length(pathes)
        try
            imgpath = joinpath(dir_path, pathes[z])
            vol[:, :, z] = load(imgpath)
        catch e
            println("Error loading image $z: $e")
            break
        end
    end
end

# Save memory usage data to a file (4st checkpoint)
open("memory_usage_checkpoint4.dat", "w") do io
    redirect_stdout(io) do
        memory_profiler.memory_usage((load_images,), interval=0.1, timeout=3600)
    end
end

# Calculate and log the estimated memory usage in GB
memory_usage_gb = prod(vol_size) * sizeof(eltype(vol)) / 1024^3
println("Estimated memory usage (GB): ", memory_usage_gb)

bin_vol = vol .> 0
count(bin_vol)
println("got past bin_vol")

point_cloud = Manifests.PointArrays.from_binary_image(bin_vol)
println("got past point_cloud")

dbf = Manifests.DBFs.compute_DBF(point_cloud, bin_vol)
println("got past first dbf")

nodeNet = Manifests.NodeNet(point_cloud; dbf=dbf)
println("finished")

using RealNeuralNetworks.NodeNets
println("now making SWC")

swc1 = NodeNets.SWCs.SWC(nodeNet)

using RealNeuralNetworks.SWCs
println("now saving SWC")

SWCs.save_swc(swc1, "plsplswork.swc")

println("Memory profiling completed. To analyze the results, follow these steps:")
println("1. Open a terminal or command prompt and navigate to the directory where this script is located.")
println("2. Run the following command to plot the memory usage:")
println("   mprof plot memory_usage.dat")
println("   This will generate a plot of the memory usage over time using matplotlib.")
println("3. To view a detailed memory usage report, run:")
println("   mprof analyze memory_usage.dat")
println("   This will provide a breakdown of the memory usage at each recorded timestamp.")
println("4. If you want to save the plot to a file instead of displaying it, run:")
println("   mprof plot memory_usage.dat --output memory_usage_plot.png")
println("   This will save the plot as an image file named 'memory_usage_plot.png'.")

