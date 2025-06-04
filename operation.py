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

def histogramGreen(image: Image.Image) -> plt:
    img_array = np.array(image)
    green_channel = img_array[:, :, 1]  # Channel hijau
    fig, ax = plt.subplots()
    ax.hist(green_channel.ravel(), bins=256, color='green', alpha=0.7)
    ax.set_title("Histogram Hijau")
    ax.set_xlim([0, 255])
    return fig

def histogramRed(image: Image.Image) -> plt:
    img_array = np.array(image)
    red_channel = img_array[:, :, 0]  # Channel merah
    fig, ax = plt.subplots()
    ax.hist(red_channel.ravel(), bins=256, color='red', alpha=0.7)
    ax.set_title("Histogram Merah")
    ax.set_xlim([0, 255])
    return fig

def histogramBlue(image: Image.Image) -> plt:
    img_array = np.array(image)
    blue_channel = img_array[:, :, 2]  # Channel biru
    fig, ax = plt.subplots()
    ax.hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.7)
    ax.set_title("Histogram Biru")
    ax.set_xlim([0, 255])
    return fig

def histogramEqu(image: Image.Image) -> plt:
    # Konversi ke array numpy
    img_array = np.array(image)
    height, width, channels = img_array.shape
    bins = 8
    bin_size = 256 // bins
    
    # Fungsi untuk equalize satu channel
    def equalize_channel(channel_data):
        # Hitung histogram
        hist = np.zeros(bins, dtype=int)
        for pixel in channel_data.ravel():
            bin_idx = pixel // bin_size
            hist[bin_idx] += 1

        # Hitung CDF
        cdf = np.cumsum(hist)
        cdf_normalized = cdf / cdf[-1]  # Normalize 0-1

        # Buat mapping nilai lama ke baru
        new_values = np.floor(cdf_normalized * 255).astype(np.uint8)
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            mapping[i] = new_values[i // bin_size]

        # Terapkan mapping ke channel
        equalized = mapping[channel_data]
        return equalized

    # Proses tiap channel RGB
    R_eq = equalize_channel(img_array[:, :, 0])
    G_eq = equalize_channel(img_array[:, :, 1])
    B_eq = equalize_channel(img_array[:, :, 2])

    # Gabungkan kembali
    equalized_img = np.stack([R_eq, G_eq, B_eq], axis=2)

    # Tampilkan gambar asli dan hasil equalization
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_array)
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(equalized_img)
    axs[1].set_title("Histogram Equalized (8 bins)")
    axs[1].axis('off')
    return fig

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def imageSpecification(image_1: Image.Image, image_2: Image.Image) -> plt:
    img1 = np.array(image_1.convert("RGB"))
    img2 = np.array(image_2.convert("RGB"))
    bins = 8
    bin_size = 256 // bins

    def match_histograms_8bin(source, reference):
        matched = np.zeros_like(source)
        for channel in range(3):  # R, G, B
            src = source[:, :, channel].ravel()
            ref = reference[:, :, channel].ravel()

            # Histogram untuk 8 bin
            src_hist = np.zeros(bins, dtype=int)
            ref_hist = np.zeros(bins, dtype=int)
            for val in src:
                src_hist[val // bin_size] += 1
            for val in ref:
                ref_hist[val // bin_size] += 1

            # Normalisasi dan CDF
            src_cdf = np.cumsum(src_hist) / len(src)
            ref_cdf = np.cumsum(ref_hist) / len(ref)

            # Mapping dari bin src ke bin ref (berbasis CDF terdekat)
            mapping = np.zeros(bins, dtype=int)
            for i in range(bins):
                diff = np.abs(ref_cdf - src_cdf[i])
                mapping[i] = np.argmin(diff)

            # Mapping dari nilai asli (0-255) ke nilai baru sesuai bin
            full_map = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                src_bin = i // bin_size
                ref_bin = mapping[src_bin]
                # Petakan ke tengah bin tujuan
                new_val = ref_bin * bin_size + bin_size // 2
                full_map[i] = np.clip(new_val, 0, 255)

            # Terapkan mapping
            matched[:, :, channel] = full_map[source[:, :, channel]]

        return matched

    # Lakukan histogram specification 8 bin
    specified_img = match_histograms_8bin(img1, img2)

    # Tampilkan
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img1)
    axs[0].set_title("Image 1 (Source)")
    axs[0].axis('off')

    axs[1].imshow(img2)
    axs[1].set_title("Image 2 (Reference)")
    axs[1].axis('off')

    axs[2].imshow(specified_img)
    axs[2].set_title("Result: Histogram Specified (8 bins)")
    axs[2].axis('off')

    return fig

