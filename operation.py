from PIL import Image
import numpy as np

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


def edge_detection(image: Image.Image, type : int) -> Image.Image:
    img_array = np.array(image, dtype=np.float32)
    
    if type == 1:  # Canny
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        kernel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float32)
    elif type == 2:  # Sobel
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        kernel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float32)
        
    elif type == 3:  # Prewitt
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)
        
        kernel_y = np.array([[ 1,  1,  1],
                            [ 0,  0,  0],
                            [-1, -1, -1]], dtype=np.float32)
    
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

    grad_x = convolve(img_array, kernel_x)
    grad_y = convolve(img_array, kernel_y)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag = (edge_mag / edge_mag.max()) * 255  # Normalize to [0,255]
    edge_img = edge_mag.astype(np.uint8)
    
    return Image.fromarray(edge_img)

def erode(image, kernel)-> Image.Image:
    kernel = np.array(kernel, dtype=np.uint8) 
    img_array = np.array(image, dtype=np.float32)
    h, w = img_array.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(img_array, pad, mode='constant', constant_values=255)
    output = np.zeros_like(img_array)

    if kernel.sum() == 0:
        return image
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.min(region[kernel == 1])
            
    return Image.fromarray(output.astype(np.uint8))

def dilation(image, kernel) -> Image.Image:
    kernel = np.array(kernel, dtype=np.uint8)
    img_array = np.array(image, dtype=np.float32)
    h, w = img_array.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(img_array, pad, mode='constant', constant_values=0)
    output = np.zeros_like(img_array)

    if kernel.sum() == 0:
        return image
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.max(region[kernel == 1])
            
    return Image.fromarray(output.astype(np.uint8))