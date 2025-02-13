from therm import get_therm_gaussian
from load import load_data
import numpy as np

loc = np.array([2, 6])
df = load_data('data/0_1.mat')
# temp00 = get_measured_temp_at_loc('00', loc)
xx, yy, g00 = get_therm_gaussian('00', df['temp_00'].mean())
# temp20 = get_measured_temp_at_loc('20', loc)
_, _, g20 = get_therm_gaussian('20', df['temp_20'].mean())
_, _, g02 = get_therm_gaussian('02', df['temp_02'].mean())
_, _, g22 = get_therm_gaussian('22', df['temp_22'].mean())
