import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================
# Función Styblinski-Tang
# =====================================================
def f(x, y):
    return 0.5 * ((x**4 - 16*x**2 + 5*x) +
                  (y**4 - 16*y**2 + 5*y))

# Gradiente
def grad(v):
    return np.array([
        2*v[0]**3 - 16*v[0] + 2.5,
        2*v[1]**3 - 16*v[1] + 2.5
    ])

# =====================================================
# Descenso por Gradiente + impresión tabla
# =====================================================
def descenso_gradiente(x0, alpha=0.01, tol=1e-4, max_iter=10000):

    xk = np.array(x0, dtype=float)
    trayectoria = [xk.copy()]
    k = 0

    print(f"\nCASO Punto inicial {x0}")
    print(f"{'k':<4}{'x_k':<25}{'∇f(x_k)':<35}{'x_{k+1}':<25}{'tolerancia'}")
    print("-"*110)

    while k < max_iter:
        g = grad(xk)
        x_next = xk - alpha*g
        error = np.linalg.norm(x_next - xk)

        print(f"{k:<4}{str(np.round(xk,6)):<25}"
              f"{str(np.round(g,6)):<35}"
              f"{str(np.round(x_next,6)):<25}"
              f"{round(error,6)}")

        trayectoria.append(x_next.copy())

        if error < tol:
            break

        xk = x_next
        k += 1

    print("Convergió en", k, "iteraciones")
    print("Mínimo aproximado:", np.round(xk,6))

    return np.array(trayectoria)

# =====================================================
# Ejecutar ambos casos
# =====================================================
tray1 = descenso_gradiente([-1, -1])
tray2 = descenso_gradiente([-4, 4])

# =====================================================
# Crear animación
# =====================================================
fig, ax = plt.subplots(figsize=(7,6))

x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax.contour(X, Y, Z, levels=30)
ax.set_title("Descenso por Gradiente - Styblinski-Tang")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# Trayectorias
line1, = ax.plot([], [], 'b-', label="Caso 1 (-1,-1)")
point1, = ax.plot([], [], 'bo')

line2, = ax.plot([], [], 'r-', label="Caso 2 (-4,4)")
point2, = ax.plot([], [], 'ro')

ax.legend()

max_frames = max(len(tray1), len(tray2))

# =====================================================
# Función de actualización (CORREGIDA)
# =====================================================
def update(frame):

    if frame < len(tray1):
        line1.set_data(tray1[:frame+1,0], tray1[:frame+1,1])
        point1.set_data([tray1[frame,0]], [tray1[frame,1]])

    if frame < len(tray2):
        line2.set_data(tray2[:frame+1,0], tray2[:frame+1,1])
        point2.set_data([tray2[frame,0]], [tray2[frame,1]])

    return line1, point1, line2, point2

ani = FuncAnimation(fig, update,
                    frames=max_frames,
                    interval=200,
                    repeat=False)

# =====================================================
# Guardar como GIF (NO requiere ffmpeg)
# =====================================================
ani.save("descenso.gif", writer="pillow")

plt.show()

print("\nAnimación guardada como: descenso_styblinski_tang.gif")