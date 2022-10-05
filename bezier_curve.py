import numpy
from matplotlib import pyplot as plt


def calculating_factorial(number: int) -> int:

    factorial = 1

    list_with_values = [j for j in range(1, number + 1)]

    for index, value in enumerate(list_with_values):
        factorial = factorial * list_with_values[index]

    return factorial


def calculating_number_of_combination(n: int, k: int, rounding=False) -> int:

    factorial_n = calculating_factorial(n)

    factorial_k = calculating_factorial(k)

    factorial_n_minus_k = calculating_factorial(n - k)

    number_of_combination = factorial_n / (factorial_k * factorial_n_minus_k)

    if rounding:

        if not isinstance(rounding, bool):
            raise ValueError

        else:

            number_of_combination = int(number_of_combination)

    return number_of_combination


def creating_bernstein_polynomial(k: int, n: int, t: numpy.ndarray):
    number_of_combination = calculating_number_of_combination(n, k, rounding=True)

    return number_of_combination * (t ** (n - k)) * (1 - t) ** k


def determination_of_coordinates(set_of_points: list) -> tuple:
    x_points = numpy.array([points[0] for points in set_of_points])
    y_points = numpy.array([points[1] for points in set_of_points])

    return x_points, y_points


def bezier_curve(set_of_points: list) -> tuple:

    number_of_points = len(set_of_points)

    all_points = determination_of_coordinates(set_of_points)

    x_points = all_points[0]
    y_points = all_points[1]

    t = numpy.linspace(0, 1, 1000)

    array_with_polynomial = numpy.array([creating_bernstein_polynomial(
        k, number_of_points - 1, t) for k in range(0, number_of_points)])

    x_axis_equation = numpy.dot(x_points, array_with_polynomial)
    y_axis_equation = numpy.dot(y_points, array_with_polynomial)

    return x_axis_equation, y_axis_equation


if __name__ == '__main__':

    set_of_points = [ [2, 1],
                      [6, 7],
                      [7, 4],
                      [9, 4] ]

    all_points = determination_of_coordinates(set_of_points)

    x_points = all_points[0]
    y_points = all_points[1]

    x_axis_equation, y_axis_equation = bezier_curve(set_of_points)

    plt.plot(x_axis_equation, y_axis_equation)
    plt.plot(x_points, y_points, "ro")

    for point_number in range(len(set_of_points)):
        plt.text(set_of_points[point_number][0], set_of_points[point_number][1],
                 (set_of_points[point_number][0], set_of_points[point_number][1]))

    plt.show()
