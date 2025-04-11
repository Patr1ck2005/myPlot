import numpy as np
import matplotlib.pyplot as plt


def generate_pattern(size, radius, back_hole_radius, width):
    """
    Generate a cross pattern with a circular hole in the center.
    """
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    theta[theta < 0] += np.pi*2

    # Create cross pattern
    cross = (theta % (np.pi/2) < width*np.pi*2).astype(float)

    cross += 10

    # Add circular hole in the center
    hole = np.clip((R/back_hole_radius)**2, 0, 1)

    hole = hole.astype(float)

    outline = R < radius

    mask = cross * hole * outline

    # return mask
    return mask

def add_vortex_phase(pattern, waist, charge):
    """
    Add vortex phase to the pattern.
    """
    size = pattern.shape[0]
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Calculate angle and add vortex phase
    R2 = X**2+Y**2
    theta = np.arctan2(Y, X)
    vortex_phase = np.exp(1j * charge * theta)
    additional_tern = 1
    return pattern * vortex_phase * additional_tern

def add_guassian(pattern, waist):
    """
    Add guassian to the pattern.
    """
    size = pattern.shape[0]
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Calculate angle and add vortex phase
    R2 = X**2+Y**2
    theta = np.arctan2(Y, X)
    gaussian_filter = np.exp(-R2/waist**2)
    return pattern * gaussian_filter

def add_spherical_phase(pattern, focal_length):
    """
    Add spherical phase to the pattern.
    """
    size = pattern.shape[0]
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Calculate angle and add vortex phase
    R2 = X**2+Y**2
    spherical_phase = np.exp(1j * focal_length * R2)
    return pattern * spherical_phase

def main():
    size = 2048  # Image size
    back_hole_radius = 0  # Radius of the circular hole
    radius = 1000  # Radius of the cross pattern
    waist = 100
    width = 0/12  # Thickness of the cross lines
    vortex_charge = 2  # Charge of the vortex phase

    # Generate pattern
    pattern = generate_pattern(size, radius, back_hole_radius, width)

    # # Add vortex phase and Gaussian Distribution
    pattern = add_guassian(pattern, waist)
    pattern_with_phase = add_vortex_phase(pattern, waist, vortex_charge)
    # pattern_with_phase = add_spherical_phase(pattern, 1e-3)
    # pattern_with_phase = pattern

    # Compute Fourier transform
    ft_pattern = np.fft.fftshift(np.fft.fft2(pattern_with_phase))

    # Visualize
    plt.figure(figsize=(18, 9))

    # Original pattern
    plt.subplot(1, 3, 1)
    plt.title("Pattern Intensity")
    original_intensity = np.abs(pattern_with_phase)**2
    plt.imshow(original_intensity, cmap="gray")
    np.save('data/fourier_applet-original_intensity', original_intensity)
    plt.axis("off")

    # Pattern with phase
    plt.subplot(1, 3, 2)
    plt.title("Pattern with Vortex Phase")
    plt.imshow(np.angle(pattern_with_phase), cmap="hsv")
    plt.axis("off")

    # Fourier transform
    plt.subplot(1, 3, 3)
    plt.title("Fourier Transform")
    # plt.imshow(np.log(1 + np.abs(ft_pattern)), cmap="inferno")
    transformed_intensity = np.abs(ft_pattern)**2
    plt.imshow(transformed_intensity, cmap="magma")
    np.save('data/fourier_applet-farfield_intensity', transformed_intensity)
    # plt.imshow(np.abs(ft_pattern)**2, cmap="twilight")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
