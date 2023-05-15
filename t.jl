import ShapeRetrieval: ShapeRetrieval as SR
using ProfileView
using BenchmarkTools

bunny = SR.load_obj("./meshes/icosahedron.obj")
# @profview SR._vertex_grad(bunny.V, bunny.F, bunny.vertex_normals);
@profview SR.load_obj("./meshes/cat.obj")
# SR.vertex_area(bunny) # Run once for precompilation
# @btime SR.vertex_area($(bunny)); # Use $ to insert variables before measurements
# @profview SR.vertex_area(bunny) # To view flame graph