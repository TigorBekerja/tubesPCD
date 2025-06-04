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

def zoomInNearestNeighbor(image: Image.Image) -> Image.Image :
    width, height = image.size
    
    image_zoom = Image.new(image.mode, (width*2, height*2))
    pix = image.load()
    pix_2 = image_zoom.load()

    for i in range(width):
        for j in range (height):
            pixel = pix[i, j]
            pix_2[i*2, j*2] = pixel
            pix_2[i*2+1, j*2] = pixel
            pix_2[i*2, j*2+1] = pixel
            pix_2[i*2+1, j*2+1] = pixel
    return image_zoom

def zoomOutNearestNeighbor(image: Image.Image) -> Image.Image :
    width, height = image.size
    
    image_zoom = Image.new(image.mode, (width//2, height//2))
    pix = image.load()
    pix_2 = image_zoom.load()

    for i in range(width//2):
        for j in range (height//2):
            pixel = pix[i*2, j*2]
            pix_2[i, j] = pixel
    return image_zoom

def zoomInBilinearInterpolation(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width, new_height = width * 2, height * 2

    image_zoom = Image.new(image.mode, (new_width, new_height))
    pix = image.load()
    pix_2 = image_zoom.load()

    for i in range(new_width):
        for j in range(new_height):
            # Koordinat (x, y) di gambar asli
            x = i / 2
            y = j / 2

            x0 = int(x)
            y0 = int(y)
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)

            dx = x - x0
            dy = y - y0

            # Ambil nilai warna dari 4 titik tetangga
            p00 = pix[x0, y0]
            p10 = pix[x1, y0]
            p01 = pix[x0, y1]
            p11 = pix[x1, y1]

            interp = tuple(
                int(round(
                    p00[k] * (1 - dx) * (1 - dy) +
                    p10[k] * dx * (1 - dy) +
                    p01[k] * (1 - dx) * dy +
                    p11[k] * dx * dy
                )) for k in range(len(p00))
            )
            pix_2[i, j] = interp
    return image_zoom

def zoomOutBilinearInterpolation(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width, new_height = width // 2, height // 2

    image_zoom = Image.new(image.mode, (new_width, new_height))
    pix = image.load()
    pix_2 = image_zoom.load()

    for i in range(new_width):
        for j in range(new_height):
            x0 = i * 2
            y0 = j * 2
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)

            p00 = pix[x0, y0]
            p10 = pix[x1, y0]
            p01 = pix[x0, y1]
            p11 = pix[x1, y1]

            avg = tuple(
                int(round((p00[k] + p10[k] + p01[k] + p11[k]) / 4))
                for k in range(len(p00))
            )
            pix_2[i, j] = avg

    return image_zoom
