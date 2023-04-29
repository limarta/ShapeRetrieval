# def cot_laplacian(vertices, faces):
#     V = vertices.shape[0]
#     F =faces.shape[0]
#     I = np.zeros(F*6, dtype=int)
#     J = np.zeros(F*6, dtype=int)
#     vals = np.zeros(F*6)
#     for shift, perm in enumerate([(0,1,2), (1,2,0), (2,0,1)]):
#         #For all 3 shifts of the roles of triangle vertices
#         #to compute different cotangent weights
#         i, j, k = perm
#         u = vertices[faces[:, i], :] - vertices[faces[:, k], :]
#         v = vertices[faces[:, j], :] - vertices[faces[:, k], :]
#         cotAlpha = np.abs(vdot(u,v)) / np.linalg.norm(np.cross(u, v, axis=1), axis=1)
#         cotAlpha = np.nan_to_num(cotAlpha,nan=0.0)
#         I[shift*F*2:shift*F*2+F] = faces[:, i]
#         J[shift*F*2:shift*F*2+F] = faces[:, j] 
#         vals[shift*F*2:shift*F*2+F] = cotAlpha
#         I[shift*F*2+F:shift*F*2+2*F] = faces[:, j]
#         J[shift*F*2+F:shift*F*2+2*F] = faces[:, i] 
#         vals[shift*F*2+F:shift*F*2+2*F] = cotAlpha
#     L = sparse.coo_matrix((vals, (I, J)), shape=(V, V)).tocsr()
#     L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape)-L
#     return L

# def heat_integrator(vertices, faces, signal, timestep=0.001, steps=100):
#     dt = timestep
#     L = cot_laplacian(vertices, faces)
#     A = vert_areas(vertices, faces)
#     M = sparse.diags(A)
#     stepOperator = (M-dt * L)
#     heat = signal.copy()
#     for i in range(steps):
#         heat = sparse.linalg.spsolve(stepOperator, heat * A)
#     return heat

# def heat_diffusion(vertices,faces,signal, t, k= 200):
#     L = cot_laplacian(vertices, faces)
#     A = vert_areas(vertices, faces)
    
#     M = sparse.diags(A)
#     L += 10*sparse.identity(vertices.shape[0])
#     eigvals, eigvecs = sparse.linalg.eigsh(L, M=M, k=k, sigma=1e-6)
#     c = (eigvecs.T @ (M@signal)) *np.exp(-t * eigvals)
#     heat = (eigvecs @ c)
#     return heat