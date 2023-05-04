
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


function get_diffusion_inputs(mesh, method::Symbol=:spectral)
    if method == :spectral
        A = mesh.vertex_area
        λ, ϕ =  get_spectrum(mesh)
        return λ, ϕ, A
    elseif method == :implicit
        return mesh.cot_laplacian, spdiagm(mesh.vertex_area), mesh.vertex_area
    end
end