import numpy as np

# Frame 19 Data
# From file (Current)
M_curr_19 = np.array([
    [0.7611561117712927, 0.32481638537670426, -0.5613694764237824, 0.33794065694358005],
    [0.06348298296363353, -0.8987076781562648, -0.43392902747206663, -0.2374169585856957],
    [-0.6454543154981943, 0.29465032260087376, -0.7046771559194867, 0.12274284276553775],
    [0, 0, 0, 1]
])
# From User (Corrected) - Transposed because Three.js matrixWorld.elements is Col-Major
M_corr_19 = np.array([
    [0.9006071671315332, 0.4316853217624466, -0.050542194724386896, 0],
    [-0.2161264547078589, 0.5456889108053697, 0.8096375534910553, 0],
    [0.3770889630420343, -0.7182418780761969, 0.5847499622643465, 0],
    [0.2762649358788634, -0.3226313090315541, 0.04985432566845627, 1]
]).T

# Frame 53 Data
# From User (Corrected)
M_corr_53 = np.array([
    [0.8697726557242523, 0.46941969755428004, -0.1521205929707368, 0],
    [-0.3149885247975588, 0.7654658807292976, 0.5611098061084799, 0],
    [0.3798391192381032,-0.44012172499580904, 0.8136431100579741, 0],
    [0.23298626313327686, 0.07048463105262387, 0.3504991575672049, 1]
]).T

# We need M_curr_53 from file as well, but for now let's just use Frame 19 to solve T.
# M_corr = T_global @ M_curr
# T_global = M_corr @ inv(M_curr)

T_fix = M_corr_19 @ np.linalg.inv(M_curr_19)

np.set_printoptions(suppress=True, precision=4)
print("Calculated T_fix based on Frame 19:")
print(T_fix)

# Check if it's a pure rotation
R_fix = T_fix[:3, :3]
det = np.linalg.det(R_fix)
print(f"Determinant: {det:.4f}")

# Check with standard axes flips
diag_flip = np.diag([1, -1, -1])
print("Is it similar to diag(1, -1, -1)?")
print(R_fix - diag_flip)

