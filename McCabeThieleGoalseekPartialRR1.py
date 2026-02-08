import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# VLE DATA
# --------------------------------------------------
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


# --------------------------------------------------
# OPERATING LINES
# --------------------------------------------------
def get_op_lines(R, xf, xd, xb, q):

    if abs(q - 1.0) < 1e-8:
        q = 1.0 - 1e-8

    # Rectifying line
    m_rect = R / (R + 1)
    c_rect = xd / (R + 1)

    # q-line
    m_q = q / (q - 1)
    c_q = -xf / (q - 1)

    # Intersection
    xi = (c_q - c_rect) / (m_rect - m_q)
    yi = m_rect * xi + c_rect

    # Stripping line
    m_strip = (yi - xb) / (xi - xb)
    c_strip = xb - m_strip * xb

    return m_rect, c_rect, m_strip, c_strip, xi, yi, m_q


# --------------------------------------------------
# THEORETICAL STAGES
# --------------------------------------------------
def count_theoretical_stages(R, xf, xd, xb, q):

    m_rect, c_rect, m_strip, c_strip, xi, yi, _ = get_op_lines(
        R, xf, xd, xb, q
    )

    pts = [(xb, xb)]

    curr_x = xb
    count = 0
    max_strip_x = xi

    while count < 100:

        curr_y = get_y_eq(curr_x)
        pts.append((curr_x, curr_y))

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


# --------------------------------------------------
# REAL TRAY MODEL (FIXED FEED TRAY)
# --------------------------------------------------
def run_real_column(nm, n_trays, R, xf, xd, xb, q):

    m_rect, c_rect, m_strip, c_strip, xi, yi, _ = get_op_lines(
        R, xf, xd, xb, q
    )

    # ---- FEED LOCATION ----
    FEED_TRAY_FROM_TOP = 6         # Feed is above Tray 7
    FEED_TRAY_FROM_BOTTOM = n_trays - FEED_TRAY_FROM_TOP

    pts = [(xb, xb)]

    curr_x = xb
    curr_y = get_y_eq(curr_x)

    pts.append((curr_x, curr_y))

    max_strip_x = xi

    for i in range(n_trays):

        # BELOW FEED → Stripping
        if i < FEED_TRAY_FROM_BOTTOM:

            curr_x = (curr_y - c_strip) / m_strip
            max_strip_x = max(max_strip_x, curr_x)

        # ABOVE FEED → Rectifying
        else:

            curr_x = (curr_y - c_rect) / m_rect

        pts.append((curr_x, curr_y))

        # Murphree efficiency correction
        y_eq = get_y_eq(curr_x)
        curr_y = curr_y + nm * (y_eq - curr_y)

        pts.append((curr_x, curr_y))

    return pts, xd - curr_y, max_strip_x


# --------------------------------------------------
# SOLVE MURPHREE EFFICIENCY
# --------------------------------------------------
def solve_murphree_eff(n_trays, R, xf, xd, xb, q):

    low = 0.01
    high = 2.0

    for _ in range(50):

        mid = (low + high) / 2

        _, error, _ = run_real_column(
            mid, n_trays, R, xf, xd, xb, q
        )

        if error < 0:
            high = mid
        else:
            low = mid

    return mid


# --------------------------------------------------
# INPUTS
# --------------------------------------------------
N_PHYSICAL_TRAYS = 8
R_VAL = 1

XF = 0.0714
XD = 0.7199
XB = 0.0112
Q = 1.146


# --------------------------------------------------
# CALCULATIONS
# --------------------------------------------------
theo_pts, N_THEO, max_theo_strip = count_theoretical_stages(
    R_VAL, XF, XD, XB, Q
)

murphree_eff = solve_murphree_eff(
    N_PHYSICAL_TRAYS, R_VAL, XF, XD, XB, Q
)

real_pts, _, max_real_strip = run_real_column(
    murphree_eff,
    N_PHYSICAL_TRAYS,
    R_VAL, XF, XD, XB, Q
)


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
plt.figure(figsize=(10, 10))


# Equilibrium + Diagonal
plt.plot(x_vle, y_vle, 'k-', lw=1.5, label='Equilibrium Curve')
plt.plot([0, 1], [0, 1], color='gray', alpha=0.3)


# Operating Lines
mr, cr, ms, cs, xi, yi, _ = get_op_lines(
    R_VAL, XF, XD, XB, Q
)

# Rectifying
plt.plot([XD, xi], [XD, yi], 'k', lw=1.2)

# Stripping
final_strip_x = max(xi, max_real_strip, max_theo_strip)

plt.plot(
    [XB, final_strip_x],
    [XB, ms * final_strip_x + cs],
    'k',
    lw=1.2,
    label='Operating Lines'
)

# q-line
plt.plot(
    [XF, xi],
    [XF, yi],
    'b-',
    lw=1.5,
    alpha=0.8,
    label=f'q-line (q={Q:.2f})'
)


# Key Points
plt.plot(XB, XB, 'ko', ms=8)
plt.text(XB, XB - 0.03, 'Xb', ha='center', fontweight='bold')

plt.plot(XF, XF, 'ko', ms=8)
plt.text(XF, XF - 0.03, 'Xf', ha='center', fontweight='bold')

plt.plot(XD, XD, 'ko', ms=8)
plt.text(XD, XD - 0.03, 'Xd', ha='center', fontweight='bold')


# Theoretical Stages
px_t, py_t = zip(*theo_pts)

plt.plot(
    px_t, py_t,
    'g--',
    lw=1.5,
    alpha=0.6,
    label=f'Theoretical: {N_THEO:.2f} stages'
)


# Real Trays
px_r, py_r = zip(*real_pts)

plt.plot(
    list(px_r) + [XD],
    list(py_r) + [py_r[-1]],
    'r-',
    lw=2,
    label='Actual Trays'
)


# --------------------------------------------------
# FEED TRAY MARKER (VERTICAL LINE)
# --------------------------------------------------
FEED_TRAY_FROM_TOP = 6
FEED_INDEX = 2 * (N_PHYSICAL_TRAYS - FEED_TRAY_FROM_TOP)

if FEED_INDEX < len(real_pts):

    x_feed = real_pts[FEED_INDEX][0]

    plt.axvline(
        x=x_feed,
        color='purple',
        linestyle='--',
        lw=2,
        alpha=0.8,
        label='Feed Tray'
    )

    plt.text(
        x_feed + 0.01,
        0.6,
        'Feed Tray\n(Above T7)',
        color='purple',
        fontweight='bold'
    )


# --------------------------------------------------
# FORMATTING
# --------------------------------------------------
plt.xlabel("Ethanol Liquid Mole Fraction (x)")
plt.ylabel("Ethanol Vapor Mole Fraction (y)")

plt.title(
    f"McCabe-Thiele | Murphree Efficiency (Emv): {murphree_eff:.2%}",
    pad=50
)

plt.legend(loc='upper left')
plt.grid(alpha=0.2)

plt.xlim(0, 0.8)
plt.ylim(0, 0.8)

# Extra top margin for cropping
plt.subplots_adjust(top=0.85)

plt.show()
