using Images, FileIO, RealNeuralNetworks
using RealNeuralNetworks.Manifests

print("it starts")

# Define the directory path and image path
dir_path = "/burg/home/jd4068/RNN/Morphology/SHL17_DownSamp"
img_path = joinpath(dir_path, "NET10_s0216.png")

# Load the first image to determine size
img = load(img_path)

# Read all image file paths in the directory
pathes = readdir(dir_path)

# Log the number of images and size of the first image
println("Number of images: ", length(pathes))
println("Size of the first image: ", size(img))

# Preallocate volume array
vol_size = (size(img)..., length(pathes))
vol = zeros(Float64, vol_size)  # Using Float64 for compatibility; adjust as needed

# Load and stack each image into the volume
for z in 1:length(pathes)
    try
        imgpath = joinpath(dir_path, pathes[z])
        vol[:, :, z] = load(imgpath)
    catch e
        println("Error loading image $z: $e")
        break
    end
end

# Calculate and log the estimated memory usage in GB
memory_usage_gb = prod(vol_size) * sizeof(eltype(vol)) / 1024^3
println("Estimated memory usage (GB): ", memory_usage_gb)


bin_vol =  vol .>0

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



