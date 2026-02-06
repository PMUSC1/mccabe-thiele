import numpy as np
import matplotlib.pyplot as plt

# --- VLE DATA ---
x_vle = np.array([0.0, 0.007, 0.015, 0.025, 0.043, 0.06, 0.105, 0.135, 0.144, 0.197, 0.274, 0.323, 0.439, 0.499, 0.577, 0.684, 0.814, 1.0])
y_vle = np.array([0.0, 0.068, 0.143, 0.206, 0.298, 0.364, 0.45, 0.478, 0.49, 0.522, 0.56, 0.579, 0.621, 0.647, 0.682, 0.739, 0.827, 1.0])

def get_y_eq(x): return np.interp(x, x_vle, y_vle)

def run_column_fixed_steps(nm, n_trays, R, xf, xd, xb, q):
    if q == 1: q -= 1e-8
    m_rect, c_rect = R/(R+1), xd/(R+1)
    m_q, c_q = q/(q-1), -xf/(q-1)
    xi = (c_q - c_rect) / (m_rect - m_q)
    yi = m_rect * xi + c_rect
    m_strip = (yi - xb) / (xi - xb)
    c_strip = xb - m_strip * xb

    pts = [(xb, xb)]
    curr_x = xb
    
    # 1. REBOILER (Stage 0) - Always 100%
    curr_y = get_y_eq(curr_x)
    pts.append((curr_x, curr_y))
    
    # 2. TRAYS (Stages 1 to N)
    # We force exactly n_trays to be calculated
    for i in range(n_trays):
        # Horizontal to Op Line
        if curr_x < xi:
            curr_x = (curr_y - c_strip) / m_strip
        else:
            curr_x = (curr_y - c_rect) / m_rect
        pts.append((curr_x, curr_y))
        
        # Vertical Efficiency Step
        y_ideal = get_y_eq(curr_x)
        curr_y = curr_y + nm * (y_ideal - curr_y)
        pts.append((curr_x, curr_y))
    
    # The final point of the staircase is the vapor of Tray 1
    # We want this curr_y to eventually equal xd
    error = xd - curr_y 
    return pts, error, (m_rect, c_rect, m_strip, c_strip, xi, yi)

def solve_efficiency(n_trays, R, xf, xd, xb, q):
    low, high = 0.01, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        _, error, _ = run_column_fixed_steps(mid, n_trays, R, xf, xd, xb, q)
        if error < 0: high = mid # Too much separation, lower efficiency
        else: low = mid          # Not enough separation, higher efficiency
    return mid

# --- INPUTS ---
N_PHYSICAL_TRAYS = 8
R_VAL = 9999  # ADJUST THIS: If Eff > 100%, increase R. If Eff is low, decrease R.
XF, XD, XB, Q = 0.0714, 0.7199, 0.0112, 1.146

eff = solve_efficiency(N_PHYSICAL_TRAYS, R_VAL, XF, XD, XB, Q)
pts, _, lines = run_column_fixed_steps(eff, N_PHYSICAL_TRAYS, R_VAL, XF, XD, XB, Q)

# --- PLOTTING ---
plt.figure(figsize=(10,10))
plt.plot(x_vle, y_vle, 'b', label='Equilibrium Curve', linewidth=2)
plt.plot([0,1],[0,1], color='gray', linestyle='--', alpha=0.3)

# Vertical Lines
plt.axvline(XB, color='purple', linestyle='--', alpha=0.5, label=f'xb={XB}')
plt.axvline(XD, color='orange', linestyle='--', alpha=0.5, label=f'xd={XD}')
plt.axvline(XF, color='green', linestyle=':', alpha=0.5, label=f'xf={XF}')

# Operating Lines
mr, cr, ms, cs, xi, yi = lines
plt.plot([XD, xi], [XD, yi], 'k', linewidth=1.2, label=f'Stripping and Rectifying Lines (R={R_VAL})')
plt.plot([XB, xi], [XB, yi], 'k', linewidth=1.2)

# Staircase - Ensure the final horizontal line hits (xd, xd)
px, py = zip(*pts)
px_final = list(px) + [XD]
py_final = list(py) + [py[-1]]
plt.plot(px_final, py_final, 'r-', linewidth=1.5, label=f'8 Trays + Reb (Eff={eff*100:.1f}%)')

# Labeling
for i in range(N_PHYSICAL_TRAYS + 1):
    idx = i * 2 + 1
    if idx < len(pts):
        label = "Reb" if i == 0 else f"T{N_PHYSICAL_TRAYS - i + 1}"
        plt.text(pts[idx][0], pts[idx][1] + 0.01, label, color='darkred', fontsize=9, ha='right')


plt.xlabel("Liquid mole fraction (x)"); plt.ylabel("Vapor mole fraction (y)")
plt.legend(loc='upper left'); plt.grid(True, alpha=0.2)
plt.xlim(0, 0.8); plt.ylim(0, 0.8); plt.show()