#! /usr/bin/env python3

import numpy as np

from differentiable_gates import Power, Unit, Multiply, Add

# implementing 2D linear regression

# load some simulated data points
data = np.loadtxt("/mnt/Data/dev/BackPropPython/data.txt")
data[:, -1] = data[:, 1] * -1

p, q = data[np.random.choice(data.shape[0])]

m_init = np.random.normal()
b_init = 0.0

x = Unit(p, 0.0)
m = Unit(m_init, 0.0)
b = Unit(b_init, 0.0)
y = Unit(-q, 0.0)

mult_gate_0 = Multiply(name="mx")
add_gate_0 = Add(name="mx+b")
add_gate_1 = Add(name="mx+b-y")
power_gate_0 = Power(p=2, name="(mx+b-y)^2")

mx = mult_gate_0.forward(m, x)
mx_b = add_gate_0.forward(mx, b)
mx_b_y = add_gate_1.forward(mx_b, y)
mx_b_y_2 = power_gate_0.forward(mx_b_y)

mx_b_y_2.gradient = 1.0
power_gate_0.backward()
add_gate_1.backward()
add_gate_0.backward()
mult_gate_0.backward()

print("x=", x)
print("m=", m)
print("b=", b)
print("y=", y)

step_size = -0.001
m.value += step_size * m.gradient
b.value += step_size * b.gradient
print("m=", m)
print("b=", b)
x.gradient = 0.0
b.gradient = 0.0
m.gradient = 0.0
y.gradient = 0.0

for i in range(0, 100):
    x.value, y.value = data[np.random.choice(data.shape[0])]

    # feed forward
    mx = mult_gate_0.forward(m, x)
    mx_b = add_gate_0.forward(mx, b)
    mx_b_y = add_gate_1.forward(mx_b, y)
    mx_b_y_2 = power_gate_0.forward(mx_b_y)

    # back propagate gradients
    mx_b_y_2.gradient = 1.0
    power_gate_0.backward()
    add_gate_1.backward()
    add_gate_0.backward()
    mult_gate_0.backward()

    # apply gradient updates to parameters
    m.value += step_size * m.gradient
    b.value += step_size * b.gradient
    print(f"[Epoch {i+1}] {m.value} {b.value}")
    # print("m=", m.value)
    # print("b=", b.valu)

    # reset gradients
    x.gradient = 0.0
    b.gradient = 0.0
    m.gradient = 0.0
    y.gradient = 0.0
