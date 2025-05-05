# Importeaza bibliotecile necesare
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp
import random

# --- Configurare ---
# Parametrii campului potential
a1 = 2.0  # Puterea respingerii
a2 = 0.5  # Decaderea razei de respingere
obstacles = [np.array([0.0, 0.0]), np.array([3.0, 4.0]), np.array([-2.0, 3.0])] # Lista centrelor obstacolelor [x, y]
p_destination = np.array([5.0, 5.0])  # Punctul destinatie [x, y]

# Parametrii simularii
T_total = 30.0  # Timpul total al simularii
dt = 0.1       # Pasul de timp pentru punctele de evaluare (solve_ivp foloseste pasi adaptivi intern)
num_agents = 3 # Numarul de agenti pentru Partea (iii)
d_inter_agent = 1.0 # Distanta minima dorita intre agenti pentru Partea (iii)
k_avoidance = 3.0 # Puterea fortei de respingere intre agenti pentru Partea (iii)

# Epsilon mic pentru a evita impartirea la zero
eps = 1e-6

# --- Functii ajutatoare ---

def attractive_potential(p, pd):
  """Calculeaza potentialul atractiv (Ec. 3)."""
  return np.linalg.norm(p - pd)

def repulsive_potential(p, ci, a1, a2):
  """Calculeaza potentialul repulsiv de la un singur obstacol (Ec. 2)."""
  dist = np.linalg.norm(p - ci)
  # Formula data are un caz special pentru p=ci.
  # Folosind formula principala a1 / (a2 + dist) gestioneaza bine apropierea pentru a2 > 0.
  if dist < eps: # Evita impartirea la zero / valori extreme daca se afla exact pe obstacol
      return a1 / a2 # Conform definitiei din text pentru p=ci
  return a1 / (a2 + dist)

def total_potential(p, pd, obstacles, a1, a2):
  """Calculeaza campul potential total P(p) (Ec. 4)."""
  potential = attractive_potential(p, pd)
  for ci in obstacles:
    potential += repulsive_potential(p, ci, a1, a2)
  return potential

def potential_gradient(p, pd, obstacles, a1, a2):
  """Calculeaza gradientul campului potential total ∇P(p) (Ec. 5)."""
  # Gradientul potentialului atractiv: (p - pd) / ||p - pd||
  dist_to_dest = np.linalg.norm(p - pd)
  if dist_to_dest < eps:
    grad_attractive = np.zeros_like(p)
  else:
    grad_attractive = (p - pd) / dist_to_dest

  # Gradientul sumei potentialelor repulsive
  grad_repulsive_sum = np.zeros_like(p)
  for ci in obstacles:
    diff = p - ci
    dist_to_obs = np.linalg.norm(diff)
    if dist_to_obs > eps: # Evita problemele de calcul daca se afla exact pe obstacol
        denominator = (a2 + dist_to_obs)**2 * dist_to_obs
        if denominator > eps: # Evita impartirea la un numar mic
             grad_repulsive_sum += -a1 * diff / denominator
        # Daca este foarte aproape, gradientul ar trebui sa fie mare si sa indice in directia opusa,
        # dar este instabil numeric. Ne bazam pe a2 pentru a preveni suprapunerea exacta.


  # Gradientul total
  grad_total = grad_attractive + grad_repulsive_sum
  return grad_total

def agent_dynamics_single(t, p, pd, obstacles, a1, a2):
    """Sistem EDO pentru un singur agent (Ec. 1 cu u din Ec. 5). dy/dt = f(t, y)"""
    # Viteza este intrarea de control u = -∇P(p)
    v = -potential_gradient(np.array(p), pd, obstacles, a1, a2)

    # Limiteaza magnitudinea vitezei pentru a preveni viteze excesive langa obstacole
    max_speed = 2.0
    speed = np.linalg.norm(v)
    if speed > max_speed:
        v = v * (max_speed / speed)

    return v.tolist()

def collision_avoidance_force(p_i, p_j, d_ij, k_avoid):
    """Calculeaza forta repulsiva asupra agentului i datorata agentului j."""
    diff = p_i - p_j
    dist = np.linalg.norm(diff)

    if dist < eps or dist >= d_ij:
        return np.zeros_like(p_i) # Fara forta daca sunt suficient de departe sau coincid

    # Forta derivata din -∇_{p_i} [ (||p_i - p_j|| - d_ij)^2 / 2 ] = - (dist - d_ij) * (p_i - p_j) / dist
    # Aceasta forta este repulsiva cand dist < d_ij
    # Folosim potentialul ζ = (∥pi − pj∥ − dij)^2 / 2
    # Forta asupra lui i este F_ij = - ∇_{pi} ζ = - (||pi - pj|| - dij) * (pi - pj) / ||pi - pj||
    force = -k_avoid * (dist - d_ij) * diff / dist

    # Optional: Foloseste un model de forta alternativ care creste mai repede la distante mici
    # force = k_avoid * (1/dist - 1/d_ij) * (1/dist**2) * diff / dist # Forma mai comuna de respingere
    # Asigura-te ca actioneaza doar cand dist < d_ij
    # force = k_avoid * max(0, (d_ij - dist)) * diff / (dist**2 + eps) # Alta abordare

    return force


def agent_dynamics_multi(t, state_flat, pd, obstacles, a1, a2, n_agents, d_ij, k_avoid):
    """Sistem EDO pentru agenti multipli cu evitarea coliziunilor."""
    state_reshaped = np.array(state_flat).reshape((n_agents, 2)) # Remodeleaza vectorul de stare plat [p1x, p1y, p2x, p2y, ...]
    velocities = np.zeros_like(state_reshaped)
    max_speed = 2.0

    for i in range(n_agents):
        p_i = state_reshaped[i]

        # 1. Forta din gradientul campului potential
        force_potential = -potential_gradient(p_i, pd, obstacles, a1, a2)

        # 2. Forta de evitare a coliziunilor cu alti agenti
        force_avoidance = np.zeros_like(p_i)
        for j in range(n_agents):
            if i == j:
                continue
            p_j = state_reshaped[j]
            force_avoidance += collision_avoidance_force(p_i, p_j, d_ij, k_avoid)

        # 3. Viteza totala (forta pentru sistemul integrator)
        v_i = force_potential + force_avoidance

        # Limiteaza magnitudinea vitezei
        speed = np.linalg.norm(v_i)
        if speed > max_speed:
            v_i = v_i * (max_speed / speed)

        velocities[i] = v_i

    return velocities.flatten().tolist() # Returneaza vitezele aplatizate

# --- Exercitiul 3 Partea (i): Construieste si ilustreaza campul potential ---
print("--- Ruleaza Exercitiul 3 Partea (i) ---")
# Creaza o grila pentru vizualizare
vis_margin = 4.0
x_range = np.arange(min(p_destination[0], min(c[0] for c in obstacles)) - vis_margin,
                    max(p_destination[0], max(c[0] for c in obstacles)) + vis_margin, 0.2)
y_range = np.arange(min(p_destination[1], min(c[1] for c in obstacles)) - vis_margin,
                    max(p_destination[1], max(c[1] for c in obstacles)) + vis_margin, 0.2)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

# Calculeaza potentialul Z pentru fiecare punct din grila
for i in range(X.shape[0]):
  for j in range(X.shape[1]):
    point = np.array([X[i, j], Y[i, j]])
    Z[i, j] = total_potential(point, p_destination, obstacles, a1, a2)

# Plotarea suprafetei 3D
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8,
                       linewidth=0, antialiased=True)
ax1.set_xlabel('Pozitia X')
ax1.set_ylabel('Pozitia Y')
ax1.set_zlabel('Potential P(p)')
ax1.set_title('Camp Potential P(p)')
fig1.colorbar(surf, shrink=0.5, aspect=5, label='Valoare Potential')

# Adauga marcatori pentru obstacole si destinatie usor deasupra suprafetei
z_offset = np.min(Z) - 0.1 # Offset pentru a face marcatorii vizibili
ax1.scatter(p_destination[0], p_destination[1], z_offset, color='lime', s=100, marker='*', label='Destinatie')
for k, obs in enumerate(obstacles):
    ax1.scatter(obs[0], obs[1], z_offset, color='red', s=80, marker='X', label=f'Obstacol {k+1}' if k == 0 else "")
ax1.legend()

# Plotarea Contururilor 2D si a Gradientului
fig2, ax2 = plt.subplots(figsize=(8, 8))
contour = ax2.contour(X, Y, Z, levels=20, cmap=cm.viridis, alpha=0.7)
fig2.colorbar(contour, label='Valoare Potential')

# Calculeaza gradientul pe grila (doar pentru vizualizare)
# Folosirea numpy.gradient este aproximativa; gradientul analitic este folosit pentru simulare
Gy, Gx = np.gradient(Z, np.mean(np.diff(y_range)), np.mean(np.diff(x_range))) # dZ/dy, dZ/dx
# Ploteaza campul gradientului negativ (-∇P) folosind quiver
skip = (slice(None, None, 3), slice(None, None, 3)) # Ploteaza mai putine sageti
ax2.quiver(X[skip], Y[skip], -Gx[skip], -Gy[skip], units='width', scale=np.percentile(np.sqrt(Gx**2+Gy**2), 95)*20, color='gray', alpha=0.8)

ax2.plot(p_destination[0], p_destination[1], 'g*', markersize=15, label='Destinatie')
for k, obs in enumerate(obstacles):
    ax2.plot(obs[0], obs[1], 'rX', markersize=12, label=f'Obstacol {k+1}' if k == 0 else "")

ax2.set_xlabel('Pozitia X')
ax2.set_ylabel('Pozitia Y')
ax2.set_title('Contururi Camp Potential si Gradient Negativ (-∇P)')
ax2.set_aspect('equal', adjustable='box')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)


# --- Exercitiul 3 Partea (ii): Simuleaza traiectoria unui singur agent ---
print("\n--- Ruleaza Exercitiul 3 Partea (ii) ---")
# Punct de start aleatoriu (evita pornirea exact pe un obstacol)
valid_start = False
while not valid_start:
    p_start_single = np.array([random.uniform(x_range[0], x_range[-1]),
                               random.uniform(y_range[0], y_range[-1])])
    valid_start = True
    for obs in obstacles:
        if np.linalg.norm(p_start_single - obs) < a2: # Verifica daca este prea aproape de centrul obstacolului
             valid_start = False
             print(f"Punctul de start {p_start_single} prea aproape de obstacolul {obs}, reincercare...")
             break

print(f"Agent unic porneste de la: {p_start_single}")

# Puncte de timp pentru evaluare
t_eval = np.arange(0, T_total, dt)

# Rezolva EDO
sol_single = solve_ivp(
    fun=agent_dynamics_single,
    t_span=[0, T_total],
    y0=p_start_single.tolist(),
    t_eval=t_eval,
    args=(p_destination, obstacles, a1, a2),
    dense_output=True # Permite plotare lina
)

# Extrage traiectoria
trajectory_single = sol_single.y

# Plotarea traiectoriei
ax2.plot(trajectory_single[0, :], trajectory_single[1, :], 'b-', linewidth=2, label='Traiectorie Agent')
ax2.plot(p_start_single[0], p_start_single[1], 'bo', markersize=8, label='Punct Start')
# Actualizeaza legenda pentru a doua figura
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # Elimina etichetele duplicate
ax2.legend(by_label.values(), by_label.keys())
fig2.suptitle("Exercitiul 3 (i) & (ii): Camp Potential si Traiectorie Agent Unic")


# --- Exercitiul 3 Partea (iii): Simuleaza agenti multipli cu evitarea coliziunilor ---
print("\n--- Ruleaza Exercitiul 3 Partea (iii) ---")

# Genereaza pozitii de start aleatorii pentru agenti multipli, asigurandu-se ca nu se suprapun initial
p_start_multi = []
min_start_dist = d_inter_agent + 0.1 # Asigura ca agentii pornesc mai departe decat d_ij
agent_count = 0
attempts = 0
max_attempts = 1000
while agent_count < num_agents and attempts < max_attempts:
    attempts += 1
    p_new = np.array([random.uniform(x_range[0], x_range[-1]),
                      random.uniform(y_range[0], y_range[-1])])

    # Verifica distanta fata de obstacole
    valid_pos = True
    for obs in obstacles:
        if np.linalg.norm(p_new - obs) < a2:
            valid_pos = False
            break
    if not valid_pos: continue

    # Verifica distanta fata de agentii deja plasati
    for p_existing in p_start_multi:
        if np.linalg.norm(p_new - p_existing) < min_start_dist:
            valid_pos = False
            break
    if not valid_pos: continue

    # Daca este valid, adauga agentul
    p_start_multi.append(p_new)
    agent_count += 1
    attempts = 0 # Reseteaza incercarile dupa plasare reusita

if agent_count < num_agents:
     print(f"Avertisment: S-au putut plasa doar {agent_count} agenti cu separare suficienta.")
     num_agents = agent_count # Ajusteaza numarul de agenti daca plasarea a esuat

if num_agents > 0:
    p_start_multi_flat = np.array(p_start_multi).flatten().tolist()
    print(f"Pozitii start multi-agent (aplatizate): {p_start_multi_flat}")

    # Rezolva sistemul EDO pentru agenti multipli
    sol_multi = solve_ivp(
        fun=agent_dynamics_multi,
        t_span=[0, T_total],
        y0=p_start_multi_flat,
        t_eval=t_eval,
        args=(p_destination, obstacles, a1, a2, num_agents, d_inter_agent, k_avoidance),
        dense_output=True
    )

    # Extrage traiectoriile
    trajectory_multi_flat = sol_multi.y
    trajectory_multi = trajectory_multi_flat.reshape((num_agents, 2, -1)) # Remodeleaza la (nr_agenti, coord_xy, pasi_timp)

    # Plotarea traiectoriilor multi-agent
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    colors = plt.cm.jet(np.linspace(0, 1, num_agents)) # Atribuie culori distincte

    # Ploteaza elementele de fundal (contururi, obstacole, destinatie)
    contour3 = ax3.contour(X, Y, Z, levels=20, cmap=cm.viridis, alpha=0.3)
    # fig3.colorbar(contour3, label='Valoare Potential') # Colorbar optional
    ax3.plot(p_destination[0], p_destination[1], 'g*', markersize=15, label='Destinatie')
    for k, obs in enumerate(obstacles):
        ax3.plot(obs[0], obs[1], 'rX', markersize=12, label=f'Obstacol {k+1}' if k == 0 else "")

    # Ploteaza traiectoria fiecarui agent si punctele de start/sfarsit
    for i in range(num_agents):
        ax3.plot(trajectory_multi[i, 0, :], trajectory_multi[i, 1, :], color=colors[i], linestyle='-', linewidth=1.5, label=f'Agent {i+1}')
        ax3.plot(trajectory_multi[i, 0, 0], trajectory_multi[i, 1, 0], 'o', color=colors[i], markersize=8, markeredgecolor='k') # Start
        ax3.plot(trajectory_multi[i, 0, -1], trajectory_multi[i, 1, -1], 's', color=colors[i], markersize=8, markeredgecolor='k') # Sfarsit

    ax3.set_xlabel('Pozitia X')
    ax3.set_ylabel('Pozitia Y')
    ax3.set_title(f'Exercitiul 3 (iii): Traiectorii {num_agents}-Agenti cu Evitarea Coliziunilor')
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend(fontsize='small')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_xlim(x_range[0], x_range[-1])
    ax3.set_ylim(y_range[0], y_range[-1])

else:
    print("Se sare peste simularea multi-agent deoarece nu s-au putut plasa agenti.")


# Afiseaza toate graficele
plt.show()

print("\n--- Terminat ---")