import os
import numpy as np
import elastica as ea





# Apply Muscle force related parameters


# Helper Function to check if a point is inside a patch
def is_point_inside_patch(point, patch_vertices):
    """
    Check if a 3D point is inside a patch defined by four vertices.

    Args:
        point (numpy.ndarray): 3D coordinates of the point.
        patch_vertices (list of numpy.ndarray): List of four 3D coordinates defining the patch.

    Returns:
        bool: True if the point is inside the patch, False otherwise.
    """
    # Calculate the normal vectors of the four triangles formed by the point and patch vertices
    normals = []
    for i in range(4):
        vertex1 = patch_vertices[i]
        vertex2 = patch_vertices[(i + 1) % 4]
        normal = np.cross(vertex2 - vertex1, point - vertex1)
        normals.append(normal)

    # Check if the point is on the same side of all the patch's triangles (using dot product)
    for i in range(4):
        if np.dot(normals[i], normals[(i + 1) % 4]) < 0:
            return False

    return True




"""
from snake_diffraction_postprocessing_3D import (
    plot_snake_velocity,
    plot_video_3d,
    compute_projected_velocity,
    plot_curvature,
)
"""

from snake_diffraction_postprocessing import (
    plot_snake_velocity,
    plot_video,
    compute_projected_velocity,
    plot_curvature,
)

class SnakeDiffractionSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    pass


def run_snake(
        b_coeff, PLOT_FIGURE=False, SAVE_FIGURE=False, SAVE_VIDEO=False, SAVE_RESULTS=False
):
    # Initialize the simulation class
    snake_diffraction_sim = SnakeDiffractionSimulator()

    # Simulation parameters
    period = 15
    final_time = (2.0 + 0.01) * period

    # setting up test params
    n_elem = 50
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    #lateral_direction = np.array([1.0,0.0,0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 0.35
    base_radius = base_length * 0.011
    density = 1000
    E = 1e6
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    snake_body = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    snake_diffraction_sim.append(snake_body)

    # Add gravitational forces
    gravitational_acc = -9.80665
    snake_diffraction_sim.add_forcing_to(snake_body).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    # Add muscle torques only Lateral Wave right now
    wave_length = b_coeff[-1]
    snake_diffraction_sim.add_forcing_to(snake_body).using(
        ea.MuscleTorques,
        base_length=base_length,
        b_coeff=b_coeff[:-1],
        period=period,
        wave_number=2.0 * np.pi / (wave_length),
        phase_shift=0.0,
        rest_lengths=snake_body.rest_lengths,
        ramp_up_time=period,
        direction=normal,
        with_spline=True,
    )

    # Add Lifting Wave 
    
    snake_diffraction_sim.add_forcing_to(snake_body).using(
      ea.LiftingTorques,
      base_length=base_length,
      b_coeff=b_coeff[:-1],
      period=period,
      wave_number=2.0 * np.pi / (wave_length),
      Amplitude=0.7,
      lamBDA=1,
      phase_shift=0.25,
      rest_lengths=snake_body.rest_lengths,
      ramp_up_time=period,
      direction=direction,
      with_spline=True,
    )
    

    # Add friction forces
    # Uniform friction with ground
    origin_plane = np.array([0.0, -base_radius, 0.0])
    normal_plane = normal
    slip_velocity_tol_Regular = 1e-6
    slip_velocity_tol_HighFriction = 1e-4
    p_factor = 30
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array_Regular = np.array(
        [mu, 2 * mu, 10.0 * mu]
    )  # [forward, backward, sideways]

    static_mu_array_Regular = np.zeros(kinetic_mu_array_Regular.shape)

    kinetic_mu_array_HighFriction = np.array(
        [mu * p_factor, 2 * mu * p_factor, 10.0 * mu * p_factor]
    )

    static_mu_array_HighFriction = np.zeros(kinetic_mu_array_HighFriction.shape)

    patch1_A_coord = np.array([0.15, 0.0, 0.6])
    patch1_B_coord = np.array([0.15, 0.0, 0.9])
    patch1_C_coord = np.array([-0.15, 0.0, 0.9])
    patch1_D_coord = np.array([-0.15, 0.0, 0.6])

    patch_vertices = [
        patch1_A_coord,
        patch1_B_coord,
        patch1_C_coord,
        patch1_D_coord,
    ]

    #print("Patch Vertices Are.")
    #print(patch_vertices)

    snake_positions = snake_body.position_collection
    #print("snake position is...")
    #print(snake_positions)
    #print("coordinates of 23 elements")
    #print(snake_positions[:, 23])

    """
    # Add frictional forces to the snake based on the region of space
    for i in range(n_elem):
        x_coord = snake_positions[0, i]  # x-coordinate of the element
        y_coord = snake_positions[1, i]  # y-coordinate of the element
        z_coord = snake_positions[2, i]  # z-coordinate of the element

        element_position = np.array([x_coord, y_coord, z_coord])  # Create a 3D position vector

        if is_point_inside_patch(element_position, patch_vertices):
            # Apply high friction forces
            mu = kinetic_mu_array_HighFriction
            static_mu = static_mu_array_HighFriction
            slip_velocity_tol = slip_velocity_tol_HighFriction
        else:
            # Apply regular friction forces
            mu = kinetic_mu_array_Regular
            static_mu = static_mu_array_Regular
            slip_velocity_tol = slip_velocity_tol_Regular

            snake_diffraction_sim.add_forcing_to(snake_body).using(
                ea.AnisotropicFrictionalPlane,
                k=1.0,
                nu=1e-6,
                plane_origin=origin_plane,
                plane_normal=normal_plane,
                slip_velocity_tol=slip_velocity_tol,
                static_mu_array=static_mu,
                kinetic_mu_array=mu,
            )
            """




    # Add uniform frictional force to snake
    snake_diffraction_sim.add_forcing_to(snake_body).using(
        ea.AnisotropicFrictionalPlane,
        k=1.0,
        nu=1e-6,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol_Regular,
        static_mu_array=static_mu_array_Regular,
        kinetic_mu_array=kinetic_mu_array_Regular,
    )


    # add damping
    damping_constant = 2e-3
    time_step = 1e-4
    snake_diffraction_sim.dampen(snake_body).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    total_steps = int(final_time / time_step)
    rendering_fps = 60
    step_skip = int(1.0 / (rendering_fps * time_step))

    # Add call backs
    class SnakeDiffractionCallBack(ea.CallBackBaseClass):
        """
        Call back function for continuum snake
        """

        def __init__(self, step_skip: int, callback_params: dict):
            ea.CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )
                self.callback_params["avg_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )

                self.callback_params["center_of_mass"].append(
                    system.compute_position_center_of_mass()
                )
                self.callback_params["curvature"].append(system.kappa.copy())

                return

    pp_list = ea.defaultdict(list)
    snake_diffraction_sim.collect_diagnostics(snake_body).using(
        SnakeDiffractionCallBack, step_skip=step_skip, callback_params=pp_list
    )

    snake_diffraction_sim.finalize()

    timestepper = ea.PositionVerlet()
    ea.integrate(timestepper, snake_diffraction_sim, final_time, total_steps)

    if PLOT_FIGURE:
        filename_plot = "snake_diffraction_velocity_test6.png"
        plot_snake_velocity(pp_list, period, filename_plot, SAVE_FIGURE)
        plot_curvature(pp_list, snake_body.rest_lengths, period, SAVE_FIGURE)

        if SAVE_VIDEO:
            filename_video = 'snake_diffraction_test7.gif'

            # filename_video = "continuum_snake.mp4"
            plot_video(
                pp_list,
                video_name=filename_video,

                xlim=(0, 4),
                ylim=(-1, 1),
            )

            #filename_video = 'snake_diffraction_3d.gif'
            #plot_video_3d(
            #    [pp_list],
            #    video_name=filename_video,

            #    fps=rendering_fps,
            #    xlim=(-3, 3),
            #    ylim=(-3, 3),
             #   zlim=(-3, 3)
            #)


    if SAVE_RESULTS:
        import pickle

        filename = "snake_diffraction_test6.dat"
        file = open(filename, "wb")
        pickle.dump(pp_list, file)
        file.close()

    # Compute the average forward velocity. These will be used for optimization.
    [_, _, avg_forward, avg_lateral] = compute_projected_velocity(pp_list, period)

    return avg_forward, avg_lateral, pp_list


if __name__ == "__main__":

    # Options
    PLOT_FIGURE = True
    SAVE_FIGURE = True
    SAVE_VIDEO = True
    SAVE_RESULTS = False
    CMA_OPTION = False

    if CMA_OPTION:
        import cma

        SAVE_OPTIMIZED_COEFFICIENTS = False


        def optimize_snake(spline_coefficient):
            [avg_forward, _, _] = run_snake(
                spline_coefficient,
                PLOT_FIGURE=False,
                SAVE_FIGURE=False,
                SAVE_VIDEO=False,
                SAVE_RESULTS=False,
            )
            return -avg_forward


        # Optimize snake for forward velocity. In cma.fmin first input is function
        # to be optimized, second input is initial guess for coefficients you are optimizing
        # for and third input is standard deviation you initially set.
        optimized_spline_coefficients = cma.fmin(optimize_snake, 7 * [0], 0.5)

        # Save the optimized coefficients to a file
        filename_data = "optimized_coefficients.txt"
        if SAVE_OPTIMIZED_COEFFICIENTS:
            assert filename_data != "", "provide a file name for coefficients"
            np.savetxt(filename_data, optimized_spline_coefficients, delimiter=",")

    else:
        # Add muscle forces on the rod
        if os.path.exists("optimized_coefficients.txt"):
            t_coeff_optimized = np.genfromtxt(
                "optimized_coefficients.txt", delimiter=","
            )
        else:
            wave_length = 1.0
            t_coeff_optimized = np.array(
                [3.4e-3, 3.3e-3, 4.2e-3, 2.6e-3, 3.6e-3, 3.5e-3]
            )
            t_coeff_optimized = np.hstack((t_coeff_optimized, wave_length))

        # run the simulation
        [avg_forward, avg_lateral, pp_list] = run_snake(
            t_coeff_optimized, PLOT_FIGURE, SAVE_FIGURE, SAVE_VIDEO, SAVE_RESULTS
        )

        print("average forward velocity:", avg_forward)
        print("average forward lateral:", avg_lateral)
