using JLD2

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


function get_diffusion_inputs(mesh::Mesh, diffusion_mode::Type{<:DiffusionMode})
    if diffusion_mode == Spectral
        A = mesh.vertex_area
        λ, ϕ =  get_spectrum(mesh)
        return λ, ϕ, convert.(Float32, A)
    else
        return mesh.cot_laplacian, mesh.vertex_area
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

function diffusion_explain(dnb::DiffusionNetBlock{Spectral}, x, λ, ϕ, A, ∇_x, ∇_y)
    x_diffused = dnb.diffusion_block(x,λ, ϕ, A)
    grad_x = ∇_x * x_diffused
    grad_y = ∇_y * x_diffused
    field = cat(grad_x, grad_y;dims=3)
    x_diffused, field, dnb(x, λ, ϕ, A, ∇_x, ∇_y)
end

OPERATOR_CACHE_FOLDER = "cached_operators"
function cached_operator_filename(mesh_name::String)
    joinpath(OPERATOR_CACHE_FOLDER, "$(mesh_name).jdl2")
end

function save_operators_to_cache(mesh::Mesh, mesh_name::AbstractString)
    filename = cached_operator_filename(mesh_name)

    cot_laplacian, vertex_area, lambda, phi, grad_x, grad_y = get_operators(mesh)
    jldsave(filename; cot_laplacian, vertex_area, lambda, phi, grad_x, grad_y)

    cot_laplacian, vertex_area, lambda, phi, grad_x, grad_y
end

function load_cached_operators(mesh_name::AbstractString)
    filename = cached_operator_filename(mesh_name)

    jldopen(filename, "r") do f
        return f["cot_laplacian"], f["vertex_area"], f["lambda"], f["phi"], f["grad_x"], f["grad_y"]
    end
end

function cache_operators_for_folder(folder::String)
    save_folder = joinpath(OPERATOR_CACHE_FOLDER, folder)
    mkpath(save_folder)
    for filepath in joinpath.(folder, readdir(folder))
        println(filepath)
        mesh = load_obj(filepath)
        mesh = normalize_mesh(mesh)
        rm(filepath)

        save_operators_to_cache(mesh, filepath)
    end
end

function load_and_cache(filepath::AbstractString)
    mesh = load_obj(filepath)
    mesh = normalize_mesh(mesh)
    rm(filepath)
    
    save_operators_to_cache(mesh, filepath) 
end