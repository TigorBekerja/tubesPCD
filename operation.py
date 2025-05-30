from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#ini buat clamping 0-255
def batasPixel(x):
    return max(0, min(255, x))

#ini buat naikin kecerahan
def modify_image_brightness(image: Image.Image, number: int) -> Image.Image:
    pix = image.load()
    width, height = image.size
    for i in range(width):
        for j in range(height):
            r, g, b, a = 0, 0, 0, 0
            if (len(pix[i, j]) == 3):
                r, g, b = pix[i, j]
            elif (len(pix[i, j]) == 4):
                r, g, b, a = pix[i, j]
            r = batasPixel(r + number)
            g = batasPixel(g + number)
            b = batasPixel(b + number)
            pix[i, j] = (r, g, b)
    return image

#ini buat contrast
def modify_image_contrast(image: Image.Image, number: int) -> Image.Image:
    pix = image.load()
    width, height = image.size
    for i in range(width):
        for j in range(height):
            r, g, b, a = 0, 0, 0, 0
            if (len(pix[i, j]) == 3):
                r, g, b = pix[i, j]
            elif (len(pix[i, j]) == 4):
                r, g, b, a = pix[i, j]
            faktor = (number/100)+1
            r = int(batasPixel(faktor*(r-128) + 128))
            g = int(batasPixel(faktor*(g-128) + 128))
            b = int(batasPixel(faktor*(b-128) + 128))
            pix[i, j] = (r, g, b)
    return image

#ini buat invert image
def modify_image_invert(image: Image.Image) -> Image.Image:
    pix = image.load()
    width, height = image.size
    for i in range(width):
        for j in range(height):
            r, g, b, a = 0, 0, 0, 0
            if (len(pix[i, j]) == 3):
                r, g, b = pix[i, j]
            elif (len(pix[i, j]) == 4):
                r, g, b, a = pix[i, j]
            r = int(batasPixel(255-r))
            g = int(batasPixel(255-g))
            b = int(batasPixel(255-b))
            pix[i, j] = (r, g, b)
    return image

def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def sobel_kernel(image: Image.Image) -> Image.Image:
    img = image.convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)
    grad_x = convolve(img_array, sobel_x)
    grad_y = convolve(img_array, sobel_y)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag = (edge_mag / edge_mag.max()) * 255  # Normalize to [0,255]
    edge_img = edge_mag.astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(edge_img, cmap='gray')
    plt.title("Sobel Edge Detection")
    plt.axis("off")
    plt.show()
    fig = plt.gcf()

    img = fig2img(fig)

    return img

def erode(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='constant', constant_values=255)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.min(region[kernel == 1])
    return output

def erosion(image: Image.Image) -> Image.Image:
    img = image.convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    grad_x = convolve(img_array, sobel_x)
    grad_y = convolve(img_array, sobel_y)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag = (edge_mag / edge_mag.max()) * 255  # Normalize to [0,255]
    edge_img = edge_mag.astype(np.uint8)

    kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
    
    
    eroded = erode(edge_img, kernel)
    plt.figure(figsize=(6, 6))
    plt.imshow(eroded, cmap='gray')
    plt.title("Erosion (+ kernel)")
    plt.axis("off")
    plt.show()
    
    fig = plt.gcf()
    img = fig2img(fig)
    return img

def dilate(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.max(region[kernel == 1])
    return output

def dilation(image: Image.Image) -> Image.Image:
    img = image.convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    grad_x = convolve(img_array, sobel_x)
    grad_y = convolve(img_array, sobel_y)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag = (edge_mag / edge_mag.max()) * 255  # Normalize to [0,255]
    edge_img = edge_mag.astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

    dilated = dilate(edge_img, kernel)

    plt.figure(figsize=(6, 6))
    plt.imshow(dilated, cmap='gray')
    plt.title("Dilation (+ kernel)")
    plt.axis("off")
    plt.show()

    fig = plt.gcf()
    img = fig2img(fig)
    return img

def opening(image: Image.Image) -> Image.Image:
    img = image.convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    grad_x = convolve(img_array, sobel_x)
    grad_y = convolve(img_array, sobel_y)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag = (edge_mag / edge_mag.max()) * 255  # Normalize to [0,255]
    edge_img = edge_mag.astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
    
    def opening(image, kernel):
        return dilate(erode(image, kernel), kernel)

    opened = opening(edge_img, kernel)

    plt.figure(figsize=(6, 6))
    plt.imshow(opened, cmap='gray')
    plt.title("Opening (Erosion → Dilation)")
    plt.axis("off")
    plt.show()
    
    fig = plt.gcf()
    img = fig2img(fig)
    return img

def closing(image: Image.Image) -> Image.Image:
    img = image.convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    grad_x = convolve(img_array, sobel_x)
    grad_y = convolve(img_array, sobel_y)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag = (edge_mag / edge_mag.max()) * 255  # Normalize to [0,255]
    edge_img = edge_mag.astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
    
    def closing(image, kernel):
        return erode(dilate(image, kernel), kernel)

    closed = closing(edge_img, kernel)

    plt.figure(figsize=(6, 6))
    plt.imshow(closed, cmap='gray')
    plt.title("Closing (Dilation → Erosion)")
    plt.axis("off")
    plt.show()

    fig = plt.gcf()
    img = fig2img(fig)
    return img