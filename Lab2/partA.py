from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv
from load import load_data
from matplotlib import pyplot as plt

df = load_data('data/0_2.mat')
df['long_dist_x'] = long_f_inv(df['long_x'])
df['medium_dist_x'] = medium_f_inv(df['med_x'])
df['short_dist_y'] = short_f_inv(df['short_y'])
df['long_dist_y'] = long_f_inv(df['long_y'])
df[['long_dist_x', 'medium_dist_x', 'short_dist_y', 'long_dist_y']].plot()
plt.show()
