import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_sqrt = 1.41421356237


def factorize(mat):
    q, r = np.linalg.qr(np.linalg.pinv(mat))
    return np.linalg.inv(r), q.T


def normalize(x):
    n = len(x)
    avg_x = np.zeros(2)

    for i in range(2):
        for j in range(n):
            avg_x[i] += x[j][i]
        avg_x[i] /= n
    # print(avg_x[0], avg_x[1])
    g_x = [[1, 0, -avg_x[0]], [0, 1, -avg_x[1]], [0, 0, 1]]
    translated_x = [transform_coordinates(x[i], g_x) for i in range(n)]
    # print(translated_x)
    avg_x_dist = 0
    for j in range(n):
        avg_x_dist += dist(translated_x[j])
    avg_x_dist /= n
    s_x = [[_sqrt / avg_x_dist, 0, 0], [0, _sqrt / avg_x_dist, 0], [0, 0, 1]]
    # print(translated_x)
    # print(s_x)
    normalized_x = np.zeros((n, 3))
    for i in range(n):
        normalized_x[i] = transform_coordinates(translated_x[i], s_x)
    # print(normalized_x)
    return normalized_x


def dist(x):
    return math.sqrt(x[0] * x[0] + x[1] * x[1])


def transform_coordinates(m, matrix):
    m_p = np.matmul(matrix, m)
    return m_p


def get_infs(x):
    a_x = np.cross(np.array(x[2]), np.array(x[1]))
    b_x = np.cross(np.array(x[6]), np.array(x[5]))
    x_left_inf = np.cross(a_x, b_x)
    a_y = np.cross(np.array(x[3]), np.array(x[2]))
    b_y = np.cross(np.array(x[7]), np.array(x[6]))
    y_left_inf = np.cross(a_y, b_y)
    a_r_x = np.cross(np.array(x[9]), np.array(x[13]))
    b_r_x = np.cross(np.array(x[10]), np.array(x[14]))
    x_right_inf = np.cross(a_r_x, b_r_x)
    a_r_y = np.cross(np.array(x[8]), np.array(x[9]))
    b_r_y = np.cross(np.array(x[11]), np.array(x[10]))
    y_right_inf = np.cross(a_r_y, b_r_y)
    return x_left_inf, y_left_inf, x_right_inf, y_right_inf


def init():
    x = [[812, 92, 1],  # X0
         [918, 138, 1],  # X1
         [761, 259, 1],  # X2
         [648, 216, 1],  # X3
         [0, 0, 1],  # X4 nevidljiva
         [904, 424, 1],  # X5
         [751, 550, 1],  # X6
         [651, 501, 1],  # X7
         [374, 442, 1],  # X0
         [691, 707, 1],  # X1
         [688, 1033, 1],  # X2
         [390, 769, 1],  # X3
         [0, 0, 1],  # X4 nevidljiva
         [1165, 465, 1],  # X5
         [1147, 782, 1],  # X6
         [0, 0, 1]]  # X7 nevidljiva

    y = [[906, 144, 1],  # X0
         [965, 205, 1],  # X1
         [741, 265, 1],  # X2
         [676, 199, 1],  # X3
         [0, 0, 1],  # X4 nevidljiva
         [933, 474, 1],  # X5
         [718, 542, 1],  # X6
         [660, 477, 1],  # X7
         [465, 356, 1],  # X0
         [572, 654, 1],  # X1
         [560, 968, 1],  # X2
         [455, 661, 1],  # X3
         [998, 297, 1],  # X4
         [1147, 580, 1],  # X5
         [1116, 887, 1],  # X6
         [0, 0, 1]  # X7 nevidljiva
         ]

    xgornje, ygornje, xdonje, ydonje = get_infs(x)
    right_xgornje, right_ygornje, right_xdonje, right_ydonje, = get_infs(y)
    print("\n\n\nINFINITIES")
    print(xgornje, ygornje, xdonje, ydonje)
    print("==============================================================\n\n")

    x[4] = np.array(np.cross(np.cross(x[7], xgornje), np.cross(x[5], ygornje)))
    x[4] = x[4] / x[4][2]
    x[12] = np.array(np.cross(np.cross(x[13], ydonje), np.cross(x[8], xdonje)))
    x[12] = x[12] / x[12][2]
    x[15] = np.array(np.cross(np.cross(x[11], xdonje), np.cross(x[14], ydonje)))
    x[15] = x[15] / x[15][2]
    print("\n\n INVISIBLE POINTS")
    print(x[4])
    print(x[12])
    print(x[15])
    y[4] = np.array(np.cross(np.cross(y[7], right_xgornje), np.cross(y[5], right_ygornje)))
    y[4] = y[4] / y[4][2]
    y[15] = np.array(np.cross(np.cross(y[11], right_xdonje), np.cross(y[14], right_ydonje)))
    y[15] = y[15] / y[15][2]
    print(y[4])
    print(y[15])
    print("==============================================================\n\n")
    return x, y


def get_fundamental(x, y):
    jed8 = np.zeros((8, 9))
    for i in range(4):
        jed8[i] = [x[i][0] * y[i][0],
                   x[i][1] * y[i][0],
                   x[i][2] * y[i][0],
                   x[i][0] * y[i][1],
                   x[i][1] * y[i][1],
                   x[i][2] * y[i][1],
                   x[i][0] * y[i][2],
                   x[i][1] * y[i][2],
                   x[i][2] * y[i][2]]
    for i in range(8,12):
        jed8[i-4] = [x[i][0] * y[i][0],
                   x[i][1] * y[i][0],
                   x[i][2] * y[i][0],
                   x[i][0] * y[i][1],
                   x[i][1] * y[i][1],
                   x[i][2] * y[i][1],
                   x[i][0] * y[i][2],
                   x[i][1] * y[i][2],
                   x[i][2] * y[i][2]]

    u, d, v = np.linalg.svd(jed8)

    f = np.zeros((3, 3))
    last_column = v[:][-1]

    for j in range(3):
        for i in range(3):
            f[j][i] = last_column[3 * j + i]
    return f


def get_epipoles(ff):
    u, d, v = np.linalg.svd(ff)
    e1 = v[2][:]
    e1 = (1 / e1[2]) * e1
    e2 = u.T[2][:]
    e2 = (1 / e2[2]) * e2
    return e1, e2


def get_camera_coordinates(mat):
    u, d, v = np.linalg.svd(mat)
    return v[:][-1]


def main():
    x, y = init()
    x_n = normalize(x)
    y_n = normalize(y)
    ff = get_fundamental(x, y)
    #ff_n = get_fundamental(x_n, y_n)
    ffx = ff
    print("FUNDAMENTALNA MATRICA:")
    print(ffx)
    # ff_n = ff_n / ff_n[2][2]
    # ff_n = -ff_n
    print("==============================================================\n\n")
    e1, e2 = get_epipoles(ffx)
    print("EPIPOLOVI FF:")
    print("epipol 1:", e1, "\nepipol 2:", e2)
    print("==============================================================\n\n")
    u, d, v = np.linalg.svd(ffx)
    d[2] = 0
    dd = [[d[0], 0, 0], [0, d[1], 0], [0, 0, 0]]
    ff1 = u @ dd @ v
    #    ff1 /= ff1[2][2]
    print("VREDNOST DETERMINANTE FF1:")
    print(np.linalg.det(ff1))
    print("==============================================================\n\n")
    print("FUNDAMENTALNA MATRICA FF1:")
    print(ff1)
    print("==============================================================\n\n")
    e1, e2 = get_epipoles(ff1)
    print("EPIPOLOVI FF1:")
    print("epipol 1:", e1, "\nepipol 2:", e2)
    print("==============================================================\n\n")
    e2_matrix = np.array([[0, -e2[2], e2[1]],
                          [e2[2], 0, -e2[0]],
                          [-e2[1], e2[0], 0]])
    print("E2 MATRICA:")
    print(e2_matrix)
    print("==============================================================\n\n")

    t1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    t2_p = e2_matrix @ ff1
    t2 = np.array([[t2_p[0][0], t2_p[0][1], t2_p[0][2], e2[0]],
                   [t2_p[1][0], t2_p[1][1], t2_p[1][2], e2[1]],
                   [t2_p[2][0], t2_p[2][1], t2_p[2][2], e2[2]]])
    print("T1 MATRICA:")
    print(t1)
    print("==============================================================\n\n")
    print("T2 MATRICA:")
    print(t2)
    print("==============================================================\n\n")

    c = get_camera_coordinates(t2)
    print("KOODRINATE DRUGE KAMERE (PRVA KAMERA 0,0,0,1):")
    print(c)
    print("==============================================================\n\n")
    k, a = factorize(t2[:, :3])
    print("MATRICA KALIBRACIJE:")
    print(k / k[2][2])
    print("==============================================================\n\n")
    print("MATRICA ORIJENTACIJE:")
    print(a)
    print("==============================================================\n\n")

    num_if_dots = len(x)
    a = np.zeros((4, 4))
    X = np.zeros((num_if_dots, 4))

    # yt1[3] - t1[2] * X = 0
    # xt1[3] + t1[1] * X = 0
    # y't2[3] - t1[2] * X = 0
    # x't1[3] + t1[1] * X = 0
    for i in range(num_if_dots):
        a[0] = x[i][1] * t1[2] - t1[1]
        a[1] = -x[i][0] * t1[2] + t1[0]
        a[2] = y[i][1] * t2[2] - t2[1]
        a[3] = -y[i][0] * t2[2] + t2[0]

        up, dp, vp = np.linalg.svd(a)
        # da bude w = 1 u X(x,y,z,w)
        X[i] = vp[:][-1] / vp[-1][-1]
        #X[i] = X[i]/X[i][2]
    # print(x)
    print("3D KOORDINATE TACAKA:")
    print(X)
    print("==============================================================\n\n")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:4, 0], X[:4, 1], X[:4, 2], color="blue")
    ax.scatter(X[4:8, 0], X[4:8, 1], X[4:8, 2], color="blue")
    ax.scatter(X[8:12, 0], X[8:12, 1], X[8:12, 2], color="red")
    ax.scatter(X[12:16, 0], X[12:16, 1], X[12:16, 2], color="red")
    plt.gca().invert_yaxis()
    plt.show()
    return


if __name__ == "__main__":
    main()
