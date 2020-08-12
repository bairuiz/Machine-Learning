import math
import numpy as np

vector = np.array([1.5, 2, 1, 2, 3])
dataset = {1: [0, 0, 1, 0, 1, 0],
           2: [0, 1, 0, 0, 0, 1],
           3: [0, 1, 1, 0, 0, 1],
           4: [1, 0, 0, 1, 0, 0]}


def phi(u):
    result = 1 / (1 + math.exp(-1 * u))
    return result


r1 = math.log(1 - phi(4))
r2 = math.log(phi(2))
r3 = math.log(phi(3))
r4 = math.log(1 - phi(3.5))
print(r1, r2, r3, r4)
print(r1 + r2 + r3 + r4)


def compute_partial_derivative(list_of_data):
    result = []
    for j in range(len(vector)):
        # print("j = ", j + 1)
        i = 0
        dot_multi = 0
        for entry in list_of_data[0:-1]:
            dot_multi += entry * vector[i]
            i += 1
        single_result = (list_of_data[-1] - phi(dot_multi)) * list_of_data[j] * -1
        result.append(single_result)
    print(result)
    return result


v0 = np.array(vector)
v1 = np.array(compute_partial_derivative(dataset[1]))
v2 = np.array(compute_partial_derivative(dataset[2]))
v3 = np.array(compute_partial_derivative(dataset[3]))
v4 = np.array(compute_partial_derivative(dataset[4]))

print(phi(3.5), phi(2) + phi(3) - 2, phi(3) + phi(4) - 1, phi(3.5), phi(4))
partial_derivate = np.array([phi(3.5), phi(2) + phi(3) - 2, phi(3) + phi(4) - 1, phi(3.5), phi(4)])
print(np.array(vector - partial_derivate))
print(phi(2.1666+0.0654))
print(phi(2.232))
