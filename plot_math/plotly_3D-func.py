import numpy as np
import plotly.graph_objects as go


def create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, a_1):
    """
    Create a paraboloid surface based on a quadratic equation.
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)

    KX, KY = np.meshgrid(kx, ky)
    omega = omega_Gamma + a_1 * (KX ** 2 + KY ** 2)  # Paraboloid equation

    return KX, KY, omega


def create_plane(kx_range, ky_range, z_height, resolution):
    """
    Create a plane at a fixed z-height.
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)

    KX, KY = np.meshgrid(kx, ky)
    Z = np.full_like(KX, z_height)

    return KX, KY, Z


def plot_surface_with_intersection(kx_range, ky_range, resolution, omega_Gamma, a_1, z_height):
    """
    Plot paraboloid and plane, and display their intersection using Plotly.
    """
    # Create paraboloid
    KX, KY, omega = create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, a_1)

    # Create plane at z_height
    PX, PY, PZ = create_plane(kx_range, ky_range, z_height, resolution)

    # Plotting the paraboloid
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=KX, y=KY, z=omega,
        colorscale='Viridis',
        opacity=0.7,
        name="Paraboloid"
    ))

    # Plotting the plane at z_height
    fig.add_trace(go.Surface(
        x=PX, y=PY, z=PZ,
        colorscale='RdBu',
        opacity=0.7,
        name="Plane at z = 1.5"
    ))

    # Customize the layout
    fig.update_layout(
        scene=dict(
            xaxis_title='kx',
            yaxis_title='ky',
            zaxis_title='omega',
            aspectmode="cube"
        ),
        title="Paraboloid and Plane Intersection",
        autosize=True
    )

    fig.show()


def main():
    # Parameters
    omega_Gamma = 1
    a_1 = 2
    kx_range = (-1, 1)  # kx range
    ky_range = (-1, 1)  # ky range
    resolution = 100  # Resolution

    # Create and visualize the surface with the intersection at z = 1.5
    plot_surface_with_intersection(kx_range, ky_range, resolution, omega_Gamma, a_1, z_height=1.5)


if __name__ == "__main__":
    main()
