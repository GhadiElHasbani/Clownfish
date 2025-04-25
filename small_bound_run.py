from model import ClownFishSchool, gaussian_neighborhood
from matplotlib import pyplot as plt


model = ClownFishSchool(n_fish=100, n_dim=3, tau=10, tau_alpha=2000, dt=0.1, n_steps=50000, seed=1, nearest_n=40, starting_alpha=0,
                        bounded=True, bounded_at=10, bound_by_reiteration=False, gamma=1,
                        use_neighborhood_function=True, neighborhood_function_params={'sigma': 10}, neighborhood_function=gaussian_neighborhood)
model.simulate(alpha_max=1.)
print(model.alpha)
for i in range(model.n_fish):
    plt.plot(model.alpha_history[:, i])
plt.xlabel("Timestep")
plt.ylabel("$\\alpha_i$")
plt.savefig('ClownFishSchool_small_alpha_history.png')
model.animate(save=True, out_file_path="ClownFishSchool_animation_small_3d.mp4", limit_axes=True)