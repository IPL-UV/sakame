from dataclasses import dataclass

# Demo Params
@dataclass
class DemoParams:
    dataset = 'moons'
    num_points = 100
    num_training = .5
    noise_level = 0.01
    plots = "demo"
    random_state = None
    n_jobs = -1
    verbose = 1
    mask_param = 1.0
    grid_points = 100


# get classification data
get_class_data(num_points=num_points,
                                        num_training=self.num_training,
                                        noise=self.noise,
                                        random_state=self.random_state,
                                        data_set=self.data_set)