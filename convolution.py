import numpy as np


def convolve(image, kernel):
    m, n, image_channels = image.shape
    kernel_size = kernel.shape[0]
    kernel_input_channels = kernel.shape[2]
    assert image_channels == kernel_input_channels
    kernel_output_channels = kernel.shape[3]
    convolved_image = np.zeros(
        (m - kernel_size + 1, n - kernel_size + 1, kernel_output_channels))
    for i in range(m - kernel_size + 1):
        for j in range(n - kernel_size + 1):
            for k in range(kernel_output_channels):
                convolved_image[i, j, k] = np.sum(
                    image[i:i+kernel_size, j:j+kernel_size, :] * kernel[:, :, :, k])
    return convolved_image


def main():
    image = np.random.random((128, 128, 3))
    kernel = np.random.random((3, 3, 3, 5))
    print(image.shape)
    print(kernel.shape)
    convolved_image = np.array(convolve(image, kernel))
    print(convolved_image.shape)


if __name__ == "__main__":
    main()
