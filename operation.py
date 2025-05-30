from PIL import Image

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