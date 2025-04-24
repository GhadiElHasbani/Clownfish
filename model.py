import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
import matplotlib as mpl
from typing import Callable, Dict


def gaussian_neighborhood(x: np.ndarray, sigma: float = 1.) -> np.ndarray:
    diffs = np.zeros((x.shape[0], x.shape[0] - 1, x.shape[1]))
    for i in range(x.shape[0]):
        diffs[i] = np.delete(x, (i), axis=0) - x[i]
    return np.sign(diffs)*np.exp(-0.5*(diffs / sigma)**2)


def parabola_gradient(a: np.ndarray, sigma: float = 1) -> np.ndarray:
    #b = np.copy(a)
    #for i in range(a.shape[1]):
        #b[:, i] = np.exp(0.5*np.sum(a ** 2 - 20, axis=1)/(sigma**2)) * a[:, i] #parabola
        #b[:, i] = -2 * a[:, i]
    #return a * (-np.exp(np.sum(-(a ** 2), axis=1)/(2 * sigma**2)) / (2*sigma**4)).reshape((-1, 1)) # gaussian
    return a * (-np.exp(0.5*np.sum(a ** 2, axis=1)/(2 * sigma**2))/(sigma**2)).reshape((-1, 1)) # parabola
    #return (-2 * a)/(sigma**2) #sphere


class ClownFishSchool:

    def __init__(self, n_fish: int = 100, n_dim: int = 3, tau: float = 100., n_steps: int = 10000, mac: bool = True, nearest_n: int = 10, dt: float = 0.1,
                 seed: int = None, tau_alpha: float = 1000., starting_alpha: float = 0.,
                 use_neighborhood_function: bool = False, neighborhood_function: Callable = None, neighborhood_function_params: Dict = None,
                 bounded: bool = False, bounded_at: float = 10, bound_by_reiteration: bool = True, gamma: float = 1):
        self.n_fish = n_fish
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.nearest_n = min([nearest_n, n_fish - 1])
        self.dt = dt

        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.bounded = bounded
        self.bounded_at = bounded_at
        self.gamma = gamma
        self.bound_by_reiteration = bound_by_reiteration

        self.fish_coords = np.random.normal(size=(n_fish, n_dim), scale=0.1)
        while np.any(np.logical_or(self.fish_coords >= np.full_like(self.fish_coords, self.bounded_at), self.fish_coords <= np.full_like(self.fish_coords, -self.bounded_at))):
            self.fish_coords = np.random.normal(size=(n_fish, n_dim), scale=0.1)
        self.fish_coords_history = np.zeros((n_steps + 1, n_fish, n_dim))
        self.alpha = np.zeros(n_fish) + starting_alpha
        self.alpha_history = np.zeros((n_steps + 1, n_fish))

        self.nearest_neighbors_ids = None
        self.use_neighborhood_function = use_neighborhood_function
        self.neighborhood_function = neighborhood_function
        self.neighborhood_function_params = neighborhood_function_params

        self.tau = tau
        self.tau_alpha = tau_alpha

        if mac:
            mpl.use('macosx')

    def get_nearest_neighbors(self):
        euclidean_distance = lambda points_a, points_b: np.sqrt(np.sum((points_a - points_b) ** 2, axis=0))
        dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.fish_coords, metric=euclidean_distance))
        nearest_neighbors_ids = np.apply_along_axis(lambda arr: arr.argpartition(range(self.nearest_n + 1))[1:self.nearest_n + 1], 1, dist_matrix)
        self.nearest_neighbors_ids = nearest_neighbors_ids

    def update_alpha(self, alpha_half: float = 3., alpha_max: float = 1., alpha_min: float = 0., alpha_slope: float = 0.3):
        #euclidean_distance = lambda points_a, points_b: np.sqrt(np.sum((points_a - points_b) ** 2, axis=0))

        #dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.fish_coords, metric=euclidean_distance))
        if self.use_neighborhood_function:
            diff_to_nearest_neighbors = np.mean(self.neighborhood_function(self.fish_coords, **self.neighborhood_function_params), axis=1)
            self.alpha += self.dt * ((alpha_max - self.alpha) * (np.linalg.norm(diff_to_nearest_neighbors, axis=1) / self.tau_alpha) + np.random.normal(size=self.n_fish, loc=0, scale=0.001))
        else:
            diff_to_nearest_neighbors = np.mean(self.fish_coords[self.nearest_neighbors_ids, :], axis=1) - self.fish_coords
            self.alpha += self.dt * ((alpha_max - self.alpha) / (np.linalg.norm(diff_to_nearest_neighbors, axis=1) * self.tau_alpha) + np.random.normal(size=self.n_fish, loc=0, scale=0.001))
        #boltzmann = lambda x: alpha_min + (alpha_max - alpha_min) / (1 + np.exp((alpha_half - x)/alpha_slope))
        #self.alpha = boltzmann(dist_matrix)
        #self.alpha += self.dt * (1 - self.alpha) / (np.mean(dist_matrix, axis=1) * self.tau_alpha)

    def update_fish_coords(self):
        scale = 1

        dfish_coords = np.random.normal(size=(self.n_fish, self.n_dim), scale=scale)  # brownian
        # alphas = np.vstack([self.alpha[i][nearest_neighbors_ids[i]] for i in range(len(self.alpha))]).reshape((len(self.alpha), -1, 1))
        # dist_to_nearest_neighbors_attraction = np.sqrt(np.sum((np.mean(alphas * self.fish_coords[nearest_neighbors_ids, :], axis=1) - self.fish_coords)**2, axis=1)).reshape((-1, 1))
        # dist_to_nearest_neighbors_repulsion = np.sqrt(np.sum((np.mean((1-alphas) * self.fish_coords[nearest_neighbors_ids, :], axis=1) - self.fish_coords) ** 2, axis=1)).reshape((-1, 1))
        if self.use_neighborhood_function:
            diff_to_nearest_neighbors = np.mean(self.neighborhood_function(self.fish_coords, **self.neighborhood_function_params), axis=1)
        else:
            diff_to_nearest_neighbors = np.mean(self.fish_coords[self.nearest_neighbors_ids, :], axis=1) - self.fish_coords
        # dist_max = 100
        # dist_half = 50
        # dist_slope = 10
        #boltzmann = lambda x: dist_max / (1 + np.exp((dist_half - x) / dist_slope))

        dfish_coords -= (1 - self.alpha).reshape((-1, 1)) * diff_to_nearest_neighbors  # repulsion
        #dfish_coords += self.alpha.reshape((-1, 1)) * diff_to_nearest_neighbors  # attraction

        if self.bounded:
            #dfish_coords += 0*(1/self.fish_coords) + 1e-03*1/(self.fish_coords - np.full_like(self.fish_coords, bounded_at)) + 1e-03*1/(self.fish_coords + np.full_like(self.fish_coords, bounded_at))
            #dfish_coords *= 1/(10*(self.fish_coords - np.full_like(self.fish_coords, bounded_at))) * 1/(10*(self.fish_coords + np.full_like(self.fish_coords, bounded_at)))

            #dfish_coords *= np.sqrt(np.sum((self.fish_coords - np.full_like(self.fish_coords, 0))**2, axis=1)).reshape((-1, 1))
            #dfish_coords /= (self.fish_coords - np.full_like(self.fish_coords, bounded_at)) * (self.fish_coords + np.full_like(self.fish_coords, bounded_at))
            if self.bound_by_reiteration:
                while np.any(np.logical_or(self.fish_coords + dfish_coords * self.dt >= np.full_like(self.fish_coords, self.bounded_at), self.fish_coords + dfish_coords * self.dt <= np.full_like(self.fish_coords, -self.bounded_at))):
                    for i in range(len(self.fish_coords)):
                        if np.any(np.logical_or(self.fish_coords[i] + dfish_coords[i] * self.dt >= np.full_like(self.fish_coords[i], self.bounded_at), self.fish_coords + dfish_coords * self.dt <= np.full_like(self.fish_coords, -self.bounded_at))):
                            dfish_coords[i] = np.random.normal(size=self.n_dim, scale=scale)
                            dfish_coords[i] -= (1 - self.alpha[i]) * (diff_to_nearest_neighbors[i])  # repulsion
                            dfish_coords[i] += self.alpha[i] * (diff_to_nearest_neighbors[i])  # attraction
            else:
                #dfish_coords += self.gamma * (np.exp(-self.fish_coords - self.bounded_at) - np.exp(self.fish_coords - self.bounded_at))
                #dfish_coords += self.gamma * (1/(1+np.exp(10*(self.fish_coords+self.bounded_at))) - 1/(1+np.exp(10*(-self.fish_coords+self.bounded_at))))
                dfish_coords += self.gamma * parabola_gradient(self.fish_coords, sigma=np.sqrt(self.bounded_at))
                #dfish_coords -= self.gamma * self.fish_coords

        self.fish_coords += dfish_coords * self.dt / self.tau

    def simulate(self, alpha_max: float = 1.):
        if self.seed is not None:
            np.random.seed(self.seed)

        for step in range(self.n_steps):
            print(f'Step: {step}')
            self.fish_coords_history[step, :, :] = self.fish_coords
            self.alpha_history[step, :] = self.alpha
            if not self.use_neighborhood_function:
                self.get_nearest_neighbors()
            self.update_alpha(alpha_max=alpha_max)
            self.update_fish_coords()

        self.alpha_history[self.n_steps, :] = self.alpha
        self.fish_coords_history[self.n_steps, :, :] = self.fish_coords

    def animate(self, save: bool = False, out_file_path: str = "clownfish_animation.mp4", limit_axes: bool = True):

        def update(frame):
            ax.cla()
            if self.n_dim == 3:
                ax.scatter(self.fish_coords_history[frame, :, 0], self.fish_coords_history[frame, :, 1], self.fish_coords_history[frame, :, 2])
                ax.text(np.max(self.fish_coords_history[:, :, 0])-0.2*np.max(self.fish_coords_history[:, :, 0]),
                        np.max(self.fish_coords_history[:, :, 1])+0.1*np.max(self.fish_coords_history[:, :, 1]),
                        z=np.max(self.fish_coords_history[:, :, 2]), s=f'frame={frame}')
                if limit_axes:
                    ax.set_zlim(np.min(self.fish_coords_history[:, :, 2]), np.max(self.fish_coords_history[:, :, 2]))
                ax.set_zlabel("$z_i$")
            else:
                ax.scatter(self.fish_coords_history[frame, :, 0], self.fish_coords_history[frame, :, 1])
                ax.text(np.max(self.fish_coords_history[:, :, 0])-0.2*np.max(self.fish_coords_history[:, :, 0]),
                        np.max(self.fish_coords_history[:, :, 1])+0.1*np.max(self.fish_coords_history[:, :, 1]), s=f'frame={frame}')

            if limit_axes:
                ax.set_xlim(np.min(self.fish_coords_history[:, :, 0]), np.max(self.fish_coords_history[:, :, 0]))
                ax.set_ylim(np.min(self.fish_coords_history[:, :, 1]), np.max(self.fish_coords_history[:, :, 1]))
            ax.set_xlabel("$x_i$")
            ax.set_ylabel("$y_i$")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if self.n_dim == 3 else None)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.n_steps + 1, interval=1)

        if save:
            print("Saving...")
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path, fps=500)

        plt.show()




