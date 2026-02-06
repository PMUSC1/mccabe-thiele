import numpy as np 
import matplotlib.pyplot as plt 

# --- EXPERIMENTAL DATA (Replace with your full table) ---
x_vle = np.array([0.0, 0.007,0.015,0.025,0.043,0.06,0.105,0.135,0.144,0.197,0.274,0.323,0.439,0.499,0.577,0.684,0.814])
y_vle = np.array([0.0, 0.068,0.143,0.206,0.298,0.364,0.45,0.478,0.49,0.522,0.56,0.579,0.621,0.647,0.682,0.739,0.827])

def get_y_eq(x, x_data, y_data):
    """Linearly interpolates VLE data to find y for a given x"""
    return np.interp(x, x_data, y_data)

def get_x_eq(y, x_data, y_data):
    """Linearly interpolates VLE data to find x for a given y (Inverse)"""
    return np.interp(y, y_data, x_data)

def calculate_step_x(y_start, x_data, y_data, efficiency):
    """Calculates the liquid composition (x) for a stage given the vapor (y),
    accounting for Murphree Efficiency."""
    x_ideal = get_x_eq(y_start, x_data, y_data)
    # Definition of efficiency relative to the equilibrium curve
    # x_actual moves only a percentage of the way towards x_ideal
    # Note: Because we step y -> x (inverse), the efficiency formula applies to the x-change.
    # Standard formula: nm = (y_n - y_n+1) / (y*_n - y_n+1). 
    # For stepping down, we simplify to linear distance scaling on the x-axis for this graphical method.
    x_actual = x_ideal # Start with ideal
    
    # We must determine the 'previous' x (which is the x on the operating line)
    # Since we don't have it passed in simply here, we use the standard efficiency approximation:
    # x_new = x_old - eff * (x_old - x_ideal). 
    # However, since we are doing a full loop, we will handle efficiency inside the main loop 
    # where we have access to 'x_old'.
    return x_ideal

def McCabeThiele_Advanced(x_vle, y_vle, R_factor, xf, xd, xb, q, nm, manual_feed_stage=None):
    """
    manual_feed_stage: Integer (e.g., 4) to force feed location. 
                       Set to None for auto-optimization.
    """
    
    # --- 1. SETUP & Q-LINE ---
    if q == 1: q -= 1e-8
    
    # Find q-line intersection with Equilibrium Curve
    xa_fine = np.linspace(0, 1, 500)
    ya_eq = get_y_eq(xa_fine, x_vle, y_vle)
    y_q_line = (q / (q - 1)) * xa_fine - (xf / (q - 1))
    
    idx = np.argwhere(np.diff(np.sign(ya_eq - y_q_line))).flatten()[0]
    q_eqX, q_eqy = xa_fine[idx], ya_eq[idx]

    # --- 2. REFLUX & OPERATING LINES ---
    # R_min calculation
    m_min = (xd - q_eqy) / (xd - q_eqX)
    R_min = m_min / (1 - m_min)
    R = R_factor * R_min
    
    # Rectifying Line (ESOL) constants: y = mx + c
    m_rect = R / (R + 1)
    c_rect = xd / (R + 1)
    
    # Intersection of ESOL and q-line (Transition point for Optimal Feed)
    m_q = q / (q - 1)
    c_q = -xf / (q - 1)
    ESOL_q_x = (c_q - c_rect) / (m_rect - m_q)
    ESOL_q_y = m_rect * ESOL_q_x + c_rect
    
    # Stripping Line (SSOL) constants
    # Slopes connects (ESOL_q_x, ESOL_q_y) and (xb, xb)
    m_strip = (xb - ESOL_q_y) / (xb - ESOL_q_x)
    c_strip = ESOL_q_y - (m_strip * ESOL_q_x)

    # --- 3. PLOTTING SETUP ---
    plt.figure(figsize=(9, 9))
    plt.plot(x_vle, y_vle, 'b-', label='Equilibrium (Table)')
    plt.plot([0,1], [0,1], 'k--', alpha=0.3)
    
    # Plot Operating Lines
    # Rectifying (Top to Feed Intersection)
    plt.plot([xd, ESOL_q_x], [xd, ESOL_q_y], 'k', linewidth=1.5, label='Rectifying Line')
    # Stripping (Feed Intersection to Bottom)
    plt.plot([xb, ESOL_q_x], [xb, ESOL_q_y], 'k', linewidth=1.5, label='Stripping Line')
    # Feed Line
    plt.plot([xf, ESOL_q_x], [xf, ESOL_q_y], 'g:', label='q-line')

    plt.axvline(xd, color='k', linestyle=':', alpha=0.5)
    plt.axvline(xb, color='k', linestyle=':', alpha=0.5)
    plt.axvline(xf, color='k', linestyle=':', alpha=0.5)

    # --- 4. STEPPING LOGIC ---
    curr_x = xd
    curr_y = xd
    stage_count = 0
    
    # We loop until we pass the bottoms composition
    while curr_x > xb:
        stage_count += 1
        
        # A. Determine which Operating Line equation to use for y
        # -----------------------------------------------------
        # If Manual: Switch if stage > manual_feed_stage
        # If Auto: Switch if x < ESOL_q_x (Intersection point)
        is_rectifying = True
        
        if manual_feed_stage is not None:
            if stage_count > manual_feed_stage:
                is_rectifying = False
        else:
            if curr_x < ESOL_q_x:
                is_rectifying = False
        
        # B. Calculate the "Ideal" Equilibrium point (x_ideal)
        # ----------------------------------------------------
        # The step starts at y = curr_y. We find x on the equilibrium curve.
        x_ideal = get_x_eq(curr_y, x_vle, y_vle)
        
        # C. Apply Efficiency & Reboiler Logic
        # ------------------------------------
        # Standard efficiency: x_actual = x_prev - nm * (x_prev - x_ideal)
        # BUT: If this step takes us past xb, it is the Reboiler.
        # The Reboiler is an equilibrium stage => 100% efficiency.
        
        # First, calculate tentative step with Tray Efficiency
        x_next_tentative = curr_x - nm * (curr_x - x_ideal)
        
        if x_next_tentative < xb:
            # We are crossing the finish line! This is the Reboiler step.
            # Force 100% efficiency (Ideal Equilibrium)
            x_next = x_ideal 
            eff_used = 1.0
            print(f"Stage {stage_count} is Reboiler (100% Eff)")
        else:
            # Normal Tray
            x_next = x_next_tentative
            eff_used = nm
        
        # D. Calculate new y on Operating Line
        # ------------------------------------
        # We draw vertical line down to x_next, then find y on the OL
        if is_rectifying:
            y_next = m_rect * x_next + c_rect
        else:
            y_next = m_strip * x_next + c_strip
            
        # E. Draw the Step
        # ----------------
        # Horizontal (Equilibrium/Efficiency step)
        plt.plot([curr_x, x_next], [curr_y, curr_y], 'r-')
        # Vertical (Operating Line step)
        # Don't draw vertical line if we are finished (below xb)
        if x_next > xb:
            plt.plot([x_next, x_next], [curr_y, y_next], 'r-')
            plt.text(x_next + 0.01, curr_y - 0.02, str(stage_count), fontsize=8)
        else:
            plt.text(x_next + 0.01, curr_y - 0.02, f"{stage_count}(Reb)", fontsize=8)

        # Update for next loop
        curr_x = x_next
        curr_y = y_next
        
        # Safety break to prevent infinite loops if data is bad
        if stage_count > 100: 
            print("Max stages reached!")
            break

    # --- 5. FINISH ---
    title_str = f"Stages: {stage_count} | Feed @ Stage: {manual_feed_stage if manual_feed_stage else 'Auto'}"
    title_str += f"\nR={round(R,2)} | Tray Eff={nm} | Reboiler Eff=1.0"
    
    plt.title(title_str)
    plt.xlabel("x (Liquid Mole Fraction)")
    plt.ylabel("y (Vapor Mole Fraction)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- INPUTS ---
# Try changing manual_feed_stage to an integer (e.g., 4) or None
McCabeThiele_Advanced(x_vle, y_vle, 
                      R_factor=1, 
                      xf=0.07143, 
                      xd=0.7184, 
                      xb=0.00595, 
                      q=1.14, 
                      nm=1, 
                      manual_feed_stage=None)