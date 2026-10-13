from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, wt_x, wt_y, wt16_x, wt16_y
from py_wake import NOJ

# Import turbine, site, and wake deficit model to use
windTurbines = V80()
site = Hornsrev1Site()
noj = NOJ(site, windTurbines)

# run model using initial positions of the wind farm

simres = noj(wt16_x, wt16_y,)

# calculate aep
print(simres.aep())

# total aep
print ("Total AEP: %f GWh"%simres.aep().sum())