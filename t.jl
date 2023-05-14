import ShapeRetrieval: ShapeRetrieval as SR
using ProfileView
using BenchmarkTools

bunny = SR.load_obj("./meshes/gourd.obj")
@profview SR.load_obj("./meshes/shrec/1.obj")
# SR.vertex_area(bunny) # Run once for precompilation
# @btime SR.vertex_area($(bunny)); # Use $ to insert variables before measurements
# @profview SR.vertex_area(bunny) # To view flame graph