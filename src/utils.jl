
# export readoff
"""
    readoff(filename)

Read a .off file and return a list of vertex positions and a triangle matrix.
"""
function readoff(filename::String)
    X, T = open(filename) do f
        s = readline(f)
        if s[1:3] != "OFF"
            error("The file is not a valid OFF one.")
        end
        s = readline(f)
        nv, nt = parse.(Int, split(s));
        X = zeros(Float64, nv, 3);
        T = zeros(Int, nt, 3);
        for i=1:nv
            s = readline(f)
            X[i,:] = parse.(Float64, split(s))
        end
        for i=1:nt
            s = readline(f)
            T[i,:] = parse.(Int64, split(s))[2:end] .+ 1
        end
        X, T
    end
end

function get_operators(mesh; k=200)
    λ, ϕ = get_spectrum(mesh, k=k)
    mesh.cot_laplacian, spdiagm(mesh.vertex_area), mesh.vertex_area, λ, ϕ, vertex_grad(mesh), vertex_grad(mesh)
end

function get_diffusion_inputs(mesh, method::Symbol=:spectral)
    if method == :spectral
        A = mesh.vertex_area
        λ, ϕ =  get_spectrum(mesh)
        return λ, ϕ, A
    elseif method == :implicit
        return mesh.cot_laplacian, spdiagm(mesh.vertex_area), mesh.vertex_area
    end
end

"""
Read OBJ files
"""
function load_obj(fname)
    verts, normals, verts_uvs, faces_verts_idx, faces_normals_idx, faces_materials_idx, material_names, mtl_path = parse_obj(fname)
    Mesh(permutedims(hcat(verts...))', permutedims(hcat(faces_verts_idx...))')
end

function parse_obj(fname; load_textures=false)
    verts, normals, verts_uvs = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
    faces_verts_idx, faces_normals_idx, faces_textures_idx = Vector{Int}[], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = nothing

    materials_idx = -1

    open(fname) do io
        for line in readlines(io)
            tokens = split(strip(line))
            if startswith(line, "mtllib")
                error("mtllib not implemented")
                if length(tokens) < 2
                    throw(ErrorException)
                end
                mtl_path = strip(line[length(tokens[1]), :])
                # TODO: Finish implementing this
            elseif length(tokens)>0 && tokens[1] == "usemtl"
                error("usemtl not implemented")
                # TODO
            elseif startswith(line, "v ") # Line is a vertex
                vert = [parse(Float64, x) for x in tokens[2:4]]
                push!(verts, vert)
            elseif startswith(line, "vt ") # Line is a texture
                tx = [parse(Float64, x) for x in tokens[2:4]]
                push!(verts_uvs, tx)
            elseif startswith(line, "vn ") # Line is a normal
                n = [parse(Float64, x) for x in tokens[2:4]]
                push!(normals, n)
            elseif startswith(line, "f ") # Line is a face
                parse_face(line, tokens, materials_idx, faces_verts_idx, faces_normals_idx, faces_textures_idx, faces_materials_idx)
            end
        end
    end
    verts, normals, verts_uvs, faces_verts_idx, faces_normals_idx, faces_materials_idx, material_names, mtl_path
    
end
function parse_face(line, tokens, materials_idx, faces_verts_idx, faces_normals_idx, faces_textures_idx, faces_materials_idx)
    face = tokens[2:end]
    face_list = [split(f, "/") for f in face]
    face_verts = Int[]
    face_normals = []
    face_textures = []
    for vert_props in face_list
        push!(face_verts, parse(Int, vert_props[1]))
        if length(vert_props) > 1
        else
        end
    end
    for i in 1:length(face_verts)-2
        # Subdivide faces with more than 3 vertices.
        push!(faces_verts_idx, [face_verts[1], face_verts[i+1], face_verts[i+2]])
    end
end