from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv
from load import load_data
from matplotlib import pyplot as plt

df = load_data('data/2_2.mat')
print(df)

df[['long_dist_x', 'medium_dist_x', 'short_dist_y', 'long_dist_y']].plot()
plt.ylim(-10, 150)
plt.show()
