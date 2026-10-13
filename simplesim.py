"""
Three-turbine wake steering simulation in PyWake (capstone starter script).

1. Defines a small model turbine (rotor diameter, hub height, power and thrust curve).
2. Places three turbines in a row, straight downwind of a fan (wind direction 270 deg).
3. For each spacing (3D, 5D, 7D):
    - computes farm efficiency at 0 deg yaw          -> PSSC 1
    - sweeps turbine 1 yaw from -30 to +30 deg       -> PSSC 3
    - sweeps turbines 1 AND 2 yaw on a grid          -> simulated ML training data (PSSC 5)
4. Saves CSV files and plots, and prints the best configuration.

Run:  python wake_steering_sim.py
Needs: pip install py_wake matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # save plots to files; remove this line to show windows instead
import matplotlib.pyplot as plt
from py_wake.site import UniformSite # Wind conditions: same speed and direction everywhere, close to fan
from py_wake.wind_turbines import WindTurbine # Turbine defiened by table of power and thrust vs wind speed
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular # ^^
from py_wake.wind_farm_models import PropagateDownwind # wind farm model that propagates wakes downwind, bending yawed wakes
from py_wake.deficit_models import BastankhahGaussianDeficit, NOJDeficit # two wake deficit models: Gaussian (better for yaw) and Jensen/Park (top-hat)
from py_wake.deflection_models import JimenezWakeDeflection # bends wake sideways wen turbines are yawed
from py_wake.superposition_models import SquaredSum # combines overlapping wakes by squaring the deficits, summing, and taking the square root
from py_wake import HorizontalGrid # grid of points to evaluate the flow field (for plotting)

# ---------------------------------------------------------------------------
# 1. SETTINGS - ##TODO: replace with measured values from the fan and turbine
# ---------------------------------------------------------------------------
D = 0.15             # rotor diameter [m]
HUB_HEIGHT = 0.30    # hub height [m]
FAN_WS = 3.0         # fan wind speed at turbine 1 [m/s], ## How to measure?
TI = 0.10            # turbulence intensity (measure or tune later) (winds gust strength / avg speed)
SPACINGS_D = [3, 5, 7]           # turbine spacings in rotor diameters
YAW_SWEEP = np.arange(-30, 31, 1)    # turbine 1 sweep [deg]
YAW_GRID = np.arange(-30, 31, 5)     # turbines 1 and 2 grid for ML data [deg]
WAKE_MODEL = "gaussian"          # "gaussian" (better for yaw) or "jensen" (Park)

# ---------------------------------------------------------------------------
# 2. TURBINE MODEL
# Power and thrust turbines produce at each wind speed
# Placeholder curves. ##TODO: Replace ws_table, power_table_W and ct_table with measured values
# ---------------------------------------------------------------------------
RHO = 1.2
AREA = np.pi * D**2 / 4 # rotor swept area [m^2]
CP_ELECTRICAL = 0.12    # rotor Cp x generator efficiency (estimate); output is electical power
CT = 0.75               # thrust coefficient (estimate); how hard rotor pushes against the wind
CUT_IN = 1.0            # [m/s]

ws_table = np.linspace(0, 10, 101) # wind speeds (0, 0.1, 0.2, ..., 10 m/s) for the power and thrust curves
# wind power formula: 0.5 * rho * A * v^3 * Cp 
# power table is wind power equation for each wind speed in ws_table, but 0 below cut-in speed
power_table_W = np.where(ws_table > CUT_IN,
                         0.5 * RHO * AREA * ws_table**3 * CP_ELECTRICAL, 0.0) 
ct_table = np.where(ws_table > CUT_IN, CT, 0.0) # below  cut0in speed of 1 m/s, power and thrust are 0

## Build the turbine model using the diameter, hub height, and power/thrust curves
turbine = WindTurbine(name="mini_turbine", diameter=D, hub_height=HUB_HEIGHT,
                      powerCtFunction=PowerCtTabular(ws_table, power_table_W, "w", ct_table))

# ---------------------------------------------------------------------------
# 3. SITE AND WIND FARM MODEL
# ---------------------------------------------------------------------------

## --- Assembles simulator --- ##
site = UniformSite(ti=TI)   # same wind everywhere, like a steady fan

# Deficit model: how much slower the air behind each turbine is
if WAKE_MODEL == "jensen":
    deficit = NOJDeficit()  # Jensen/Park top-hat model (wake decay k inside)
else:
    deficit = BastankhahGaussianDeficit(use_effective_ws=True)

wfm = PropagateDownwind(site, turbine,
                        wake_deficitModel=deficit,
                        superpositionModel=SquaredSum(), # handles a turbine sitting in more than one wake
                        deflectionModel=JimenezWakeDeflection())  # bends yawed wakes
## --- --- #

WD = 270  # wind direction ##TODO: implement varying wind directions for training
# current setting: wind from the west, blowing along +x, straight down the row

# --- Helper function to run the model for a given spacing and yaw angles --- #
def farm_power(spacing_D, yaws):
    """Return power per turbine [W] for a row at the given spacing and yaw angles."""
    x = np.arange(3) * spacing_D * D # Turbine positions along the x-axis (downwind)
    y = np.zeros(3) # Turbine positions along the y-axis (no offset)
    res = wfm(x, y, wd=WD, ws=FAN_WS, yaw=np.asarray(yaws, float), tilt=0) # runs simulaiton
    # tilt = 0; rotors arent tilted up or down, just yawed left/right
    return res.Power.values.squeeze() # pulls power value as array of three values (one per turbine)


# Single turbine alone = no wake losses (denominator for efficiency)
p_alone = farm_power(1000, [0, 0, 0])[0]   # huge spacing -> no wake interaction
print(f"Single turbine alone at {FAN_WS} m/s: {p_alone*1000:.2f} mW\n")

# ---------------------------------------------------------------------------
# 4. RUN THE STUDY
# ---------------------------------------------------------------------------
sweep_rows, grid_rows, summary_rows = [], [], []

for s in SPACINGS_D:
    # Baseline: everyone faces the wind
    p0 = farm_power(s, [0, 0, 0])
    eff0 = p0.sum() / (3 * p_alone)

    # Sweep turbine 1 only
    for g in YAW_SWEEP:
        p = farm_power(s, [g, 0, 0])
        sweep_rows.append(dict(spacing_D=s, yaw1=g, yaw2=0, yaw3=0,
                               P1_mW=p[0]*1e3, P2_mW=p[1]*1e3, P3_mW=p[2]*1e3,
                               Ptotal_mW=p.sum()*1e3))

    # Grid over turbines 1 and 2 (turbine 3 stays at 0: yawing the last one never helps)
    for g1 in YAW_GRID:
        for g2 in YAW_GRID:
            p = farm_power(s, [g1, g2, 0])
            grid_rows.append(dict(spacing_D=s, fan_ws=FAN_WS, yaw1=g1, yaw2=g2, yaw3=0,
                                  P1_mW=p[0]*1e3, P2_mW=p[1]*1e3, P3_mW=p[2]*1e3,
                                  Ptotal_mW=p.sum()*1e3))

    df_s = pd.DataFrame([r for r in sweep_rows if r["spacing_D"] == s])
    best = df_s.loc[df_s.Ptotal_mW.idxmax()]
    summary_rows.append(dict(spacing_D=s,
                             efficiency_0deg=eff0,
                             Ptotal_0deg_mW=p0.sum()*1e3,
                             best_yaw1_deg=best.yaw1,
                             Ptotal_best_mW=best.Ptotal_mW,
                             gain_percent=(best.Ptotal_mW / (p0.sum()*1e3) - 1) * 100))

sweep = pd.DataFrame(sweep_rows)
grid = pd.DataFrame(grid_rows)
summary = pd.DataFrame(summary_rows)

sweep.to_csv("yaw1_sweep.csv", index=False)
grid.to_csv("ml_training_data_sim.csv", index=False)
summary.to_csv("summary.csv", index=False)

# Rank the 6 PSSC 3 configurations: 3 spacings x turbine 1 at 0 deg or best yaw
configs = []
for _, r in summary.iterrows():
    configs.append((f"{r.spacing_D:.0f}D, yaw1 = 0", r.Ptotal_0deg_mW))
    configs.append((f"{r.spacing_D:.0f}D, yaw1 = {r.best_yaw1_deg:.0f}", r.Ptotal_best_mW))
configs.sort(key=lambda c: -c[1])

pd.set_option("display.precision", 3)
print("Summary per spacing (PSSC 1 efficiency, PSSC 3 best yaw):")
print(summary.to_string(index=False))
print("\nBest turbine 1 + 2 combination (from the grid):")
for s in SPACINGS_D:
    g = grid[grid.spacing_D == s]
    b = g.loc[g.Ptotal_mW.idxmax()]
    base = g[(g.yaw1 == 0) & (g.yaw2 == 0)].Ptotal_mW.item()
    print(f"  {s}D: yaw1 = {b.yaw1:.0f}, yaw2 = {b.yaw2:.0f} -> {b.Ptotal_mW:.2f} mW "
          f"({(b.Ptotal_mW/base - 1)*100:+.1f}% vs 0/0)")

print("\nPSSC 3 ranking of configurations (predicted):")
for i, (name, p) in enumerate(configs, 1):
    print(f"  {i}. {name:<20} {p:7.2f} mW")

# ---------------------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for s in SPACINGS_D:
    d = sweep[sweep.spacing_D == s]
    ax.plot(d.yaw1, d.Ptotal_mW, label=f"{s}D spacing")
ax.set_xlabel("Turbine 1 yaw angle [deg]")
ax.set_ylabel("Total farm power [mW]")
ax.set_title(f"Simulated wake steering ({WAKE_MODEL} wake model, {FAN_WS} m/s)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("farm_power_vs_yaw.png", dpi=150)

# Flow maps at 5D: no yaw vs best yaw
s = 5
best_yaw = summary.loc[summary.spacing_D == s, "best_yaw1_deg"].item()
x = np.arange(3) * s * D
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5))
for ax, g in zip(axes, [0, best_yaw]):
    res = wfm(x, np.zeros(3), wd=WD, ws=FAN_WS, yaw=np.array([g, 0, 0], float), tilt=0)
    fm = res.flow_map(HorizontalGrid(x=np.linspace(-2*D, x[-1] + 5*D, 300),
                                     y=np.linspace(-2*D, 2*D, 120)))
    fm.plot_wake_map(ax=ax)
    ax.set_title(f"5D spacing, turbine 1 yaw = {g:.0f} deg, "
                 f"total = {res.Power.values.sum()*1e3:.1f} mW")
fig.tight_layout()
fig.savefig("flow_map_5D.png", dpi=150)

print("\nSaved: yaw1_sweep.csv, ml_training_data_sim.csv, summary.csv,")
print("       farm_power_vs_yaw.png, flow_map_5D.png")