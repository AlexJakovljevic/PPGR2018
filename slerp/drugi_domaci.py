import math as m
import numpy as np

e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def euler_2_a(phi, theta, psi):
    rotation_matrix_x = [[1, 0, 0], [0, m.cos(phi), -m.sin(phi)], [0, m.sin(phi), m.cos(phi)]]
    rotation_matrix_y = [[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]]
    rotation_matrix_z = [[m.cos(psi), -m.sin(psi), 0], [m.sin(psi), m.cos(psi), 0], [0, 0, 1]]

    return np.matmul(np.matmul(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)


def get_submatrix(a, submatrix, i, j):
    n = len(a[0])
    i_submatrix = 0
    j_submatrix = 0
    for i_matrix in range(n):
        for j_matrix in range(n):
            if i_matrix != i and j_matrix != j:
                submatrix[i_submatrix][j_submatrix] = a[i_matrix][j_matrix]
                j_submatrix += 1
                if j_submatrix == n - 1:
                    j_submatrix = 0
                    i_submatrix += 1

    return submatrix


def determinant(a):
    n = len(a[0])
    if n == 1:
        return a[0][0]
    submatrix = np.zeros((n - 1, n - 1))
    det = 0
    for j in range(n):
        submatrix = get_submatrix(a, submatrix, 0, j)
        minor = determinant(submatrix)
        if j % 2 == 0:
            sign = 1
        else:
            sign = -1
        kofaktor = minor * sign
        det += kofaktor * a[0][j]

    return det


def check_a_2_angle_axis(a):
    for i in range(3):
        for j in range(3):
            if a[i][j] != e[i][j]:
                return False

    if round(determinant(a), 2) == 1:
        return False
    return True


def magnitude(v):
    return m.sqrt(sum([v[i] ** 2 for i in range(len(v))]))


def a_2_angle_axis(a):
    # proveravamo da li je detA = 1 i da li je A =/ E
    if check_a_2_angle_axis(a):
        print("INVALID MATRIX")
        exit(1)
    ae = a - e
    p = np.cross(ae[0], ae[1])
    p_norm = [i/magnitude(p) for i in p]
    u = np.array(ae[1])
    u_p = np.matmul(a, np.transpose(u))
    phi = m.acos(np.dot(u, u_p)/(magnitude(u)*magnitude(u_p)))
    if determinant([u, u_p, p_norm]) < 0:
        p_norm = [-i for i in p_norm]
    return p_norm, phi


def rodrigues(p, phi):  # Benjamin Olinde Rodrigues
    p_x = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    p = np.array([[p[0], p[1], p[2]]])
    ppt = np.matmul(np.transpose(p), p)
    rodrigues_ = ppt + m.cos(phi) * (e - ppt) + m.sin(phi) * p_x
    return rodrigues_


def a_2_euler(a):
    if check_a_2_angle_axis(a):
        print("INVALID MATRIX!")
        exit(1)

    if a[2][0] < 1:
        if a[2][0] > -1:
            psi = m.atan2(a[1][0], a[0][0])
            theta = m.asin(-a[2][0])
            phi = m.atan2(a[2][1], a[2][2])
        else:
            psi = m.atan2(-a[0][1], a[1][1])
            theta = m.pi / 2
            phi = 0
    else:
        psi = m.atan2(-a[0][1], a[1][1])
        theta = -m.pi / 2
        phi = 0

    return phi, theta, psi


def axis_angle_2_q(p, phi):
    if phi == 0:
        return [0, 0, 0, 1]
    w = m.cos(phi / 2)

    q = [m.sin(phi / 2) * p[i] for i in range(len(p))]
    q.append(w)
    return q


def q_2_axis_angle(q):
    n = len(q)
    q = [q[i] / magnitude(q) for i in range(n)]
    if q[3] < 0:
        q = [-q[i] for i in range(n)]

    w = q[3]
    phi = 2 * m.acos(w)
    if m.fabs(w) == 1:
        p = [1, 0, 0]
    else:
        p = [q[i] / magnitude(q[:3]) for i in range(3)]
    return p, phi


def lerp(q1, q2, tm, t):
    k1 = 1 - t / tm
    k2 = t / tm
    q1 = [i * k1 for i in q1]
    q2 = [i * k2 for i in q2]
    qt = [i + j for i, j in zip(q1, q2)]
    n = m.sqrt(sum(i ** 2 for i in qt))
    return [i / n for i in qt]


def slerp(q1, q2, tm, t):
    if tm < t < 0:
        exit(1)

    cos_0 = sum(i * j for i, j in zip(q1, q2))
    if cos_0 < 0:
        q1 = [-i for i in q1]
        cos_0 = -cos_0

    if cos_0 > 0.95:
        return lerp(q1, q2, tm, t)

    phi_0 = m.acos(cos_0)
    k1 = m.sin(phi_0 * (1 - t / tm) / m.sin(phi_0))
    k2 = m.sin(phi_0 * t / tm) / m.sin(phi_0)
    q1 = [(k1 * i) for i in q1]
    q2 = [(k2 * i) for i in q2]

    return [i + j for i, j in zip(q1, q2)]


def interpolation(t1, t2, t):
    t1 = [(1 - t) * i for i in t1]
    t2 = [t * i for i in t2]
    return [t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2]]


def main():
    print("=========== PRVI DEO ZADATKA ===========")
    # Uzete su konstantne vrednosti za uglove zbog originalnog test primera. Inace, samo treba promeniti u input().
    phi = m.radians(43)
    theta = m.radians(73)
    psi = m.radians(-15)
    print("Angles:")
    print(phi, theta, psi)
    print("Euler2A")
    a_test = euler_2_a(phi, theta, psi)
    print(a_test)
    print("A2AxisAngle")
    p_test, phi_test = a_2_angle_axis(a_test)
    print(p_test, phi_test)
    print("Rodrigues")
    a_rodrigues = rodrigues(p_test, phi_test)
    print(a_rodrigues)
    print("A2Euler")
    a_rodrigues_2euler = a_2_euler(a_rodrigues)
    print(a_rodrigues_2euler)
    print("AxisAngle2Q")
    q = axis_angle_2_q(p_test, phi_test)
    print(q)
    print("Q2AxisAngle")
    print(q_2_axis_angle(q))
    print("=========== DRUGI DEO ZADATKA ===========")
    print("Unesite tri koordinate pocetne tacke i tri ugla u stepenima, svaki unos u novom redu: ")
    dot_1 = [float(input()) for _ in range(3)]
    print("Uglovi:")
    phi_1 = m.radians(float(input()))
    theta_1 = m.radians(float(input()))
    psi_1 = m.radians(float(input()))
    print("Unesite tri koordinate krajnje tacke i tri ugla u stepenima, svaki unos u novom redu: ")
    dot_2 = [float(input()) for _ in range(3)]
    print("Uglovi:")
    phi_2 = m.radians(float(input()))
    theta_2 = m.radians(float(input()))
    psi_2 = m.radians(float(input()))

    a1 = euler_2_a(phi_1, theta_1, psi_1)
    a2 = euler_2_a(phi_2, theta_2, psi_2)

    begin_v, begin_a = a_2_angle_axis(a1)
    end_v, end_a = a_2_angle_axis(a2)

    q1 = axis_angle_2_q(begin_v, begin_a)
    q2 = axis_angle_2_q(end_v, end_a)

    # Ispis u fajl za program koji pokrece animaciju u C-u
    # with open("animation_file.txt", "w") as f:
    #     f.write(str(dot_1[0]) + " " + str(dot_1[1]) + " " + str(dot_1[2]) + "\n")
    #     f.write(str(dot_2[0]) + " " + str(dot_2[1]) + " " + str(dot_2[2]) + "\n")
    #     p1, phi_1 = q_2_axis_angle(q1)
    #     p2, phi_2 = q_2_axis_angle(q2)
    #     f.write(str(p1[0]) + " " + str(p1[1]) + " " + str(p1[2]) + " " + str(phi_1) + "\n")
    #     f.write(str(p2[0]) + " " + str(p2[1]) + " " + str(p2[2]) + " " + str(phi_2) + "\n")
    #     tm = 1.0
    #     for t in np.arange(0.0, tm, 0.01):
    #         val = slerp(q1, q2, tm, t)
    #         p, phi = q_2_axis_angle([val[0], val[1], val[2], val[3]])
    #         dot_t = interpolation(dot_1, dot_2, t)
    #
    #         print(p, round(phi, 2), round(dot_t[0], 2), round(dot_t[1], 2), round(dot_t[2], 2))
    #         f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) +
    #                 " " + str(phi) + " " + str(dot_t[0]) + " " + str(dot_t[1]) + " " + str(dot_t[2]) + "\n")
    #

    print("q1:", q1)
    print("q2:", q2)
    tm = 1.0
    for t in np.arange(0.0, tm, 0.006):
        val = slerp(q1, q2, tm, t)
        p, phi = q_2_axis_angle([val[0], val[1], val[2], val[3]])
        dot_t = interpolation(dot_1, dot_2, t)

        print("P:", round(p[0], 2), round(p[1], 2), round(p[2], 2), "\tphi:", round(phi, 2), "\tdot:",
              round(dot_t[0], 2), round(dot_t[1], 2),
              round(dot_t[2], 2))


if __name__ == "__main__":
    main()
