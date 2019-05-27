#! /usr/bin/env python3

import math


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def cross_entropy(x, y):
    eps = 1e-8
    return y * math.log(x+eps) + (1 - y) * math.log(1-x+eps)


def zeros_initializer(shape=(1,)):
    pass


class Initializer(object):

    def __init__(self, function):
        self.initialize = function

    def __call__(self, *args, **kwargs):
        return self.initialize(*args, **kwargs)


class Unit(object):

    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

    def __str__(self):
        return f"V: {self.value} - G: {self.gradient}"


class Gate(object):

    def __init__(self, name):
        self.name = name

    def forward(self, input_1, input_2):
        self.unit_0 = input_1
        self.unit_1 = input_2
        self.utop = Unit(0.0, 0.0)
        return self.utop

    def backward(self):
        self.unit_0.gradient += 0.0
        self.unit_1.gradient += 0.0

    def __str__(self):
        return f"{self.utop.value}"


class Multiply(Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_1, input_2):
        self.unit_0 = input_1
        self.unit_1 = input_2
        self.utop = Unit(input_1.value * input_2.value, 0.0)
        return self.utop

    def backward(self):
        self.unit_0.gradient += self.unit_1.value * self.utop.gradient
        self.unit_1.gradient += self.unit_0.value * self.utop.gradient


class Add(Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_1, input_2):
        self.unit_0 = input_1
        self.unit_1 = input_2
        self.utop = Unit(input_1.value + input_2.value, 0.0)
        return self.utop

    def backward(self):
        self.unit_0.gradient += 1.0 * self.utop.gradient
        self.unit_1.gradient += 1.0 * self.utop.gradient


class Sigmoid(Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_1):
        self.unit_0 = input_1
        self.utop = Unit(sigmoid(input_1.value), 0.0)
        return self.utop

    def backward(self):
        x = sigmoid(self.unit_0.value)
        self.unit_0.gradient += x * (1 - x) * self.utop.gradient


class CrossEntropy(Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_1, label):
        self.unit_0 = input_1
        self.utop = Unit(cross_entropy(input_1.value, label), 0.0)
        return self.utop

    def backward(self, y):
        z = self.unit_0.value + 1e-8
        self.unit_0.gradient += ((y/z) - ((1-y)/(1-z))) * self.utop.gradient


def main():

    # Create input units
    a = Unit(1.0, 0.0)
    b = Unit(2.0, 0.0)
    c = Unit(-3.0, 0.0)
    x = Unit(-1.0, 0.0)
    y = Unit(3.0, 0.0)

    label = 1

    print(a)
    print(b)
    print(c)
    print(x)
    print(y)

    # Create the gates for function:
    # f(z) = sigmoid(z), z = ax + by + c

    mult_gate_0 = Multiply("a*x")
    mult_gate_1 = Multiply("b*y")

    add_gate_0 = Add("a*x + b*y")
    add_gate_1 = Add("(a*x + b*y) + c")
    sig_gate_0 = Sigmoid("sigmoid(z)")

    xent_gate_0 = CrossEntropy("Xent(s, y)")

    ax = mult_gate_0.forward(a, x)
    by = mult_gate_1.forward(b, y)
    ax_by = add_gate_0.forward(ax, by)
    z = add_gate_1.forward(ax_by, c)
    s = sig_gate_0.forward(z)
    ce = xent_gate_0.forward(s, label)

    step_size = 0.01

    for i in range(0, 1000):

        ce.gradient = 1.0
        xent_gate_0.backward(label)
        sig_gate_0.backward()
        add_gate_1.backward()
        add_gate_0.backward()
        mult_gate_1.backward()
        mult_gate_0.backward()

        a.value += step_size * a.gradient
        b.value += step_size * b.gradient
        c.value += step_size * c.gradient
        x.value += step_size * x.gradient
        y.value += step_size * y.gradient

        print("")
        print(a)
        print(b)
        print(c)
        print(x)
        print(y)
        print("")

        print(s, ce)
        ax = mult_gate_0.forward(a, x)
        by = mult_gate_1.forward(b, y)
        ax_by = add_gate_0.forward(ax, by)
        z = add_gate_1.forward(ax_by, c)
        s = sig_gate_0.forward(z)
        ce = xent_gate_0.forward(s, label)
        print(s, ce)


if __name__ == "__main__":
    main()
