import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

_sqrt = 1.41421356237


# Aleksandar Jakovljevic, 156/15

def main():
    A = [-4, -2, 1]
    B = [-3, -1, 1]
    C = [-2, 2, 1]
    D = [-3, 4, 1]
    E = [-6, 3, 1]
    F = [-5, 1, 1]

    Ap = [1, -2, 1]
    Bp = [3, -3, 1]
    Cp = [5, -1.5, 1]
    Dp = [4, 3, 1]
    Ep = [3.5, 1, 1]
    Fp = [2, 0, 1]
    M = [-2.5, 0, 1]
    originals = [A, B, C, D, E, F]
    images = [Ap, Bp, Cp, Dp, Ep, Fp]
    originals_x = [original[0] for original in originals]
    originals_y = [original[1] for original in originals]
    originals_x.append(A[0])
    originals_y.append(A[1])
    images_x = [image[0] for image in images]
    images_y = [image[1] for image in images]
    images_x.append(Ap[0])
    images_y.append(Ap[1])
    originals_naive_x = originals_x[0:4]
    originals_naive_y = originals_y[0:4]

    # print("Naive algorithm:")
    # print(naive_algorithm(originals[0:4], images[0:4]))
    # print("DLT for four:")
    # print(simple_dlt(originals[0:4], images[0:4]))
    # print("Scaled DLT for four:")
    # print(np.multiply(naive_algorithm(originals, images)[0][0]/simple_dlt(originals[0:4], images[0:4])[0][0],simple_dlt(originals[0:4], images[0:4])))
    # print("Normalized DLT for four:")
    # print(normalized_dlt(originals[0:4], images[0:4]))
    # print("Scaled normalized DLT for four:")
    # print(np.multiply(naive_algorithm(originals, images)[0][0]/normalized_dlt(originals[0:4], images[0:4])[0][0],normalized_dlt(originals[0:4], images[0:4])))
    print("Naive algorithm:")
    print(naive_algorithm(originals, images))
    print("DLT:")
    print(simple_dlt(originals, images))
    print("Normalized DLT for six:")
    print(normalized_dlt(originals, images))
    print("Scaled normalized DLT for six:")
    print(np.multiply(simple_dlt(originals, images)[0][0] / normalized_dlt(originals, images)[0][0],
                      normalized_dlt(originals, images)))
    # print(transform_coordinates(M, naive_algorithm(originals, images)))
    # print(transform_coordinates(M, simple_dlt(originals, images)))
    # print(transform_coordinates(M, normalized_dlt(originals, images)))
    M_ = transform_coordinates(M, naive_algorithm(originals, images))
    M_dlt = transform_coordinates(M, simple_dlt(originals, images))
    M_ndlt = transform_coordinates(M, normalized_dlt(originals, images))
    originals_naive_x = originals_x[0:4]
    originals_naive_y = originals_y[0:4]
    originals_naive_x.append(A[0])
    originals_naive_y.append(A[1])
    plt.plot(originals_naive_x, originals_naive_y, 'bo')
    plt.plot(originals_naive_x, originals_naive_y, 'black')
    plt.plot(originals_x, originals_y, 'ro')
    plt.plot(originals_x, originals_y, 'red')
    plt.plot(images_x, images_y, 'go')
    plt.plot(images_x, images_y, 'green')
    plt.axis([-10, 10, -10, 10])
    plt.plot([M[0], M_[0], M_dlt[0], M_ndlt[0]], [M[1], M_[1], M_dlt[1], M_ndlt[1]], 'bo')

    plt.plot([M_[0], M_ndlt[0], M_dlt[0]], [M_[1], M_ndlt[1], M_dlt[1]], 'black')
    plt.annotate('input', xy=(M[0], M[1]), xytext=(M[0] - 1, M[1] + 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.annotate('naive', xy=(M_[0], M_[1]), xytext=(M_[0] - 1, M_[1] + 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.annotate('dlt', xy=(M_dlt[0], M_dlt[1]), xytext=(M_dlt[0] - 1, M_dlt[1] + 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.annotate('ndlt', xy=(M_ndlt[0], M_ndlt[1]), xytext=(M_ndlt[0] - 1, M_ndlt[1] + 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

    return


def normalized_dlt(x, x_p):
    n = len(x)
    avg_x = np.zeros(2)
    avg_x_p = np.zeros(2)

    for i in range(2):
        for j in range(n):
            avg_x_p[i] += x_p[j][i]
            avg_x[i] += x[j][i]
        avg_x_p[i] /= n
        avg_x[i] /= n

    G_x = [[1, 0, -avg_x[0]], [0, 1, -avg_x[1]], [0, 0, 1]]
    G_x_p = [[1, 0, -avg_x_p[0]], [0, 1, -avg_x_p[1]], [0, 0, 1]]
    translated_x = [transform_coordinates(x[i], G_x) for i in range(n)]
    translated_x_p = [transform_coordinates(x_p[i], G_x_p) for i in range(n)]

    avg_x_dist = 0
    avg_x_p_dist = 0

    for j in range(n):
        avg_x_p_dist += dist(translated_x_p[j])
        avg_x_dist += dist(translated_x[j])

    avg_x_p_dist /= n
    avg_x_dist /= n

    S_x = np.matmul(np.linalg.inv(G_x),
                    np.matmul([[_sqrt / avg_x_dist, 0, 0], [0, _sqrt / avg_x_dist, 0], [0, 0, 1]], G_x))
    T_x = np.matmul(S_x, G_x)

    S_x_p = np.matmul(np.linalg.inv(G_x),
                      np.matmul([[_sqrt / avg_x_p_dist, 0, 0], [0, _sqrt / avg_x_p_dist, 0], [0, 0, 1]], G_x))
    T_x_p = np.matmul(S_x_p, G_x_p)
    normalized_x = np.zeros((n, 3))
    normalized_x_p = np.zeros((n, 3))
    for i in range(n):
        normalized_x[i] = transform_coordinates(x[i], T_x)
        normalized_x_p[i] = transform_coordinates(x_p[i], T_x_p)

    P = simple_dlt(normalized_x, normalized_x_p)
    P_ = np.matmul(np.matmul(np.linalg.inv(T_x_p), P), T_x)

    return P_


def dist(x):
    return math.sqrt(x[0] * x[0] + x[1] * x[1])


def simple_dlt(x, x_p):
    n = len(x)
    A = np.zeros((2 * n, 9))

    for i in range(n):
        A[2 * i] = [0, 0, 0,
                    -x_p[i][2] * x[i][0], -x_p[i][2] * x[i][1], -x_p[i][2] * x[i][2],
                    x_p[i][1] * x[i][0], x_p[i][1] * x[i][1], x_p[i][1] * x[i][2]]
        A[2 * i + 1] = [x_p[i][2] * x[i][0], x_p[i][2] * x[i][1], x_p[i][2] * x[i][2],
                        0, 0, 0,
                        -x_p[i][0] * x[i][0], -x_p[i][0] * x[i][1], -x_p[i][0] * x[i][2]]

    # print(A)
    U, D, V = np.linalg.svd(A)
    matrix = np.zeros((3, 3))
    # print(V[:][-1])
    last_column = V[:][-1]
    for j in range(3):
        for i in range(3):
            matrix[j][i] = last_column[3 * j + i]

    return matrix


def transform_coordinates(M, matrix):
    M_p = np.matmul(matrix, M)
    return M_p


def naive_algorithm(originals, images):
    A = originals[0]
    B = originals[1]
    C = originals[2]
    D = originals[3]
    A_p = images[0]
    B_p = images[1]
    C_p = images[2]
    D_p = images[3]
    lambdas = find_lambdas(A, B, C, D)
    P_1 = find_matrix(A, B, C, lambdas)
    # print(P_1)

    lambdas_p = find_lambdas(A_p, B_p, C_p, D_p)
    P_2 = find_matrix(A_p, B_p, C_p, lambdas_p)
    # print(P_2)

    transformation_matrix = np.matmul(P_2, np.linalg.inv(P_1))

    return transformation_matrix


def find_matrix(A, B, C, lambdas):
    P = np.array([[A[0] * lambdas[0], B[0] * lambdas[1], C[0] * lambdas[2]],
                  [A[1] * lambdas[0], B[1] * lambdas[1], C[1] * lambdas[2]],
                  [A[2] * lambdas[0], B[2] * lambdas[1], C[2] * lambdas[2]]])

    return P


def find_lambdas(A, B, C, D):
    left_side = np.array([[A[0], B[0], C[0]],
                          [A[1], B[1], C[1]],
                          [A[2], B[2], C[2]]])
    right_side = np.array(D)
    lambdas = np.linalg.solve(left_side, right_side)

    return lambdas


if __name__ == '__main__':
    main()
