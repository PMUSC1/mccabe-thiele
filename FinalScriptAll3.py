import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==================================================
# VLE DATA
# ==================================================
x_vle = np.array([
    0.0, 0.007, 0.015, 0.025, 0.043, 0.06, 0.105, 0.135,
    0.144, 0.197, 0.274, 0.323, 0.439, 0.499, 0.577,
    0.684, 0.814, 1.0
])

y_vle = np.array([
    0.0, 0.068, 0.143, 0.206, 0.298, 0.364, 0.45, 0.478,
    0.49, 0.522, 0.56, 0.579, 0.621, 0.647, 0.682,
    0.739, 0.827, 1.0
])

def get_y_eq(x):
    return np.interp(x, x_vle, y_vle)

# ==================================================
# OPERATING LINES
# ==================================================
def get_op_lines(R, xf, xd, xb, q):
    
    # Avoid division by zero for q=1
    if abs(q - 1.0) < 1e-8:
        q = 1.0 - 1e-8

    m_rect = R / (R + 1)
    c_rect = xd / (R + 1)

    m_q = q / (q - 1)
    c_q = -xf / (q - 1)

    xi = (c_q - c_rect) / (m_rect - m_q)
    yi = m_rect * xi + c_rect

    m_strip = (yi - xb) / (xi - xb)
    c_strip = xb - m_strip * xb

    return m_rect, c_rect, m_strip, c_strip, xi, yi

# ==================================================
# THEORETICAL STAGES
# ==================================================
def count_theoretical_stages(R, xf, xd, xb, q):

    m_rect, c_rect, m_strip, c_strip, xi, yi = get_op_lines(
        R, xf, xd, xb, q
    )

    pts = [(xb, xb)]
    curr_x = xb

    count = 0
    max_strip_x = xi

    while count < 100:
        curr_y = get_y_eq(curr_x)
        pts.append((curr_x, curr_y))

        # Optimal switching: Switch when vapor y passes intersection yi
        if curr_y < yi:
            next_x = (curr_y - c_strip) / m_strip
            max_strip_x = max(max_strip_x, next_x)
        else:
            next_x = (curr_y - c_rect) / m_rect

        if next_x > xd:
            count += (xd - curr_x) / (next_x - curr_x)
            pts.append((xd, curr_y))
            break

        pts.append((next_x, curr_y))
        curr_x = next_x
        count += 1

        if curr_x >= xd:
            break

    return pts, count, max_strip_x

# ==================================================
# REAL COLUMN MODEL (OPTIMAL FEED)
# ==================================================
def run_real_column(nm, n_trays, R, xf, xd, xb, q):

    m_rect, c_rect, m_strip, c_strip, xi, yi = get_op_lines(
        R, xf, xd, xb, q
    )

    pts = [(xb, xb)]

    curr_x = xb
    curr_y = get_y_eq(curr_x)

    pts.append((curr_x, curr_y))

    max_strip_x = xi

    for i in range(n_trays):
        
        # --- OPTIMAL FEED LOGIC ---
        # If current vapor y is below intersection yi, use Stripping line.
        # Otherwise, switch to Rectifying line.
        if curr_y < yi:
            curr_x = (curr_y - c_strip) / m_strip
            max_strip_x = max(max_strip_x, curr_x)
        else:
            curr_x = (curr_y - c_rect) / m_rect

        pts.append((curr_x, curr_y))

        # Apply Murphree Efficiency
        y_eq = get_y_eq(curr_x)
        curr_y = curr_y + nm * (y_eq - curr_y)

        pts.append((curr_x, curr_y))

    return pts, xd - curr_y, max_strip_x

# ==================================================
# SOLVE MURPHREE EFFICIENCY
# ==================================================
def solve_murphree_eff(n_trays, R, xf, xd, xb, q):
    low = 0.01
    high = 2.0  # Allow > 100% just in case

    for _ in range(50):
        mid = (low + high) / 2
        _, error, _ = run_real_column(
            mid, n_trays, R, xf, xd, xb, q
        )
        # If error > 0, we haven't reached Xd yet (efficiency too low)
        if error > 0:
            low = mid
        else:
            high = mid

    return mid

# ==================================================
# SINGLE CASE PLOT FUNCTION
# ==================================================
def run_case(case_num, n_trays, R, xf, xd, xb, q):

    # ---- Calculations ----
    theo_pts, N_theo, max_theo = count_theoretical_stages(
        R, xf, xd, xb, q
    )

    murphree = solve_murphree_eff(
        n_trays, R, xf, xd, xb, q
    )

    real_pts, _, max_real = run_real_column(
        murphree, n_trays, R, xf, xd, xb, q
    )

    # ---- Plot ----
    plt.figure(figsize=(9, 9))
    

    # Equilibrium (Purple)
    plt.plot(x_vle, y_vle, 'tab:purple', lw=2, label='Equilibrium') 
    plt.plot([0, 1], [0, 1], color='gray', alpha=0.3)

    # Operating lines
    mr, cr, ms, cs, xi, yi = get_op_lines(
        R, xf, xd, xb, q
    )

    plt.plot([xd, xi], [xd, yi], 'k', lw=1.2)

    final_strip = max(xi, max_real, max_theo)

    plt.plot(
        [xb, final_strip],
        [xb, ms * final_strip + cs],
        'k',
        lw=1.2,
        label='Operating Lines'
    )

    # q-line (Conditional Label)
    q_label = f'q = {q:.2f}'
    if abs(q - 1.0) < 1e-6:
        q_label = None

    plt.plot(
        [xf, xi],
        [xf, yi],
        'b-',
        lw=1.5,
        alpha=0.8,
        label=q_label
    )

    # Points
    plt.plot(xb, xb, 'ko', ms=7)
    plt.plot(xf, xf, 'ko', ms=7)
    plt.plot(xd, xd, 'ko', ms=7)

    # -- LABELS --
    # Xb to the RIGHT (+0.02)
    plt.text(xb + 0.02, xb, 'Xb', ha='left', va='center')
    
    plt.text(xf, xf - 0.03, 'Xf', ha='center', va='top')
    plt.text(xd, xd - 0.03, 'Xd', ha='center', va='top')

    # Theoretical
    px_t, py_t = zip(*theo_pts)
    plt.plot(
        px_t, py_t,
        'g--',
        lw=1.3,
        alpha=0.6,
        label=f'Theo: {N_theo:.2f}'
    )

    # Real trays
    px_r, py_r = zip(*real_pts)
    plt.plot(
        list(px_r) + [xd],
        list(py_r) + [py_r[-1]],
        'r-',
        lw=2,
        label='Actual'
    )

    # Formatting
    plt.xlabel("Ethanol Liquid Mole Fraction (x)")
    plt.ylabel("Ethanol Vapor Mole Fraction (y)")

    plt.title(
        f"Case {case_num} | Emv = {murphree:.2%}",
        pad=45
    )

    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.8)
    plt.subplots_adjust(top=0.85)

    plt.show()

# ==================================================
# DEFINE YOUR THREE CASES HERE
# ==================================================
cases = [
    # (N trays, R, xf, xd, xb, q)
    # (8, 1, 0.0714, 0.7199, 0.0112, 1.146),
    # (8, 1, 0.0702, 0.7009, 0.006633,  1.141),
    # (8, 999, 0.0714, 0.7184,   0.00595,  1)
    #(N trays, R, xf, xd, xb, q)

    # #partial 1
    # (8, 1, 0.0714, 0.7145, 0.0112, 1.146),
    # (8, 1, 0.0714, 0.7215, 0.0112,  1.146),
    # (8, 1, 0.0714, 0.7238,   0.0112,  1.146)

    # #partial 2
    # (8, 1, 0.0702, 0.6811, 0.00709, 1.141),
    # (8, 1, 0.0702, 0.6854, 0.00640,  1.141),
    # (8, 1, 0.0702, 0.7009,   0.0064,  1.141)

    #Total
    (8, 999, 0.0714, 0.7145, 0.00368, 1),
    (8, 999, 0.0714, 0.7122, 0.00709,  1),
    (8, 999, 0.0714, 0.729,   0.00709,  1)
]

# ==================================================
# RUN ALL CASES
# ==================================================
for i, case in enumerate(cases, start=1):
    run_case(
        i,
        case[0],   # trays
        case[1],   # R
        case[2],   # xf
        case[3],   # xd
        case[4],   # xb
        case[5]    # q
    )