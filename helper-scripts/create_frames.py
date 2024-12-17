import cv2
import os
import numpy as np
import random
import qrcode 
import string
import random
import multiprocessing as mp
from PIL import Image

def create_image(params):
    background = params

    
    characters = string.ascii_letters + string.digits + string.punctuation
    crop_size = 512
    bg_height, bg_width = background.shape[:2]


    random_string = ''.join(random.choices(characters, k=random.randint(4, 64)))
    code = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )

    code.add_data(random_string)
    code.make(fit=True)

    qr = code.make_image(fill_color="black", back_color="white")
    pil_data = qr.convert('RGB')

    qr = np.array(pil_data)[:, :, ::-1].copy()

    d_width = random.randint(96, 412)

    qr = cv2.resize(qr, (d_width, d_width), interpolation= cv2.INTER_AREA)

    angle = random.uniform(-11.5, 11.5)
    # Get the dimensions of the image
    (h, w) = qr.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    qr = cv2.warpAffine(qr, rotation_matrix, (w, h), borderValue=(255, 255, 255))

    # crop the image to 512 x 512
    x = random.randint(0, bg_width - crop_size)
    y = random.randint(0, bg_height - crop_size)

    cropped_image = background[y:y + crop_size, x:x + crop_size]
    qr_height, qr_width = qr.shape[:2]

    x = random.randint(0, crop_size - qr_width)
    y = random.randint(0, crop_size - qr_height)

    positive_frame = cropped_image.copy()
    negative_frame = cropped_image.copy()
    epsilon = random.randint(10, 30)
    black_image = cropped_image.copy()
    black_image = np.zeros((512,512, 3))

    for y1 in range(0,  cropped_image.shape[0]):
        for x1 in range(0,  cropped_image.shape[1]):
            rl = epsilon
            gl = epsilon
            bl = epsilon

            r, g,b = cropped_image[y1][x1]

            r = (r * (235) / 255) + 10
            g = (g * (235) / 255) + 10
            b = (b * (235) / 255) + 10

            if(r + rl > 255): 
                rl = 255-r
            if(g + gl > 255): 
                gl = 255-g
            if(b + bl > 255): 
                bl = 255-b


            if(r - rl < 0):
                rl = 0+r
            if(g - gl < 0):
                gl = 0+g
            if(b - bl < 0):
                bl = 0+b

            positive_frame[y1][x1] = (r+rl, g+gl, b+bl)
            negative_frame[y1][x1] = (r-rl, g-gl, b-bl)


    for y1 in range(0,qr_height):
        for x1 in range(0,qr_width):

            rl = epsilon
            gl = epsilon
            bl = epsilon

            r, g,b = cropped_image[y1 + y][x1 + x]

            r = (r * (235) / 255) + 10
            g = (g * (235) / 255) + 10
            b = (b * (235) / 255) + 10

            if(r + rl > 255): 
                rl = 255-r
            if(g + gl > 255): 
                gl = 255-g
            if(b + bl > 255): 
                bl = 255-b


            if(r - rl < 0):
                rl = 0+r
            if(g - gl < 0):
                gl = 0+g
            if(b - bl < 0):
                bl = 0+b

            if np.sum(qr[y1][x1]) < 240:
                positive_frame[y1 + y][x1 + x] = (r-rl, g-gl, b-bl)
                negative_frame[y1 + y][x1 + x] = (r+rl, g+gl, b+bl)
            else:
                positive_frame[y1 + y][x1 + x] = (r+rl, g+gl, b+bl)
                negative_frame[y1 + y][x1 + x] = (r-rl, g-gl, b-bl)

            if np.sum(qr[y1][x1]) < 240:
                black_image[y1 + y][x1 + x] = 255

    return positive_frame, negative_frame, black_image
                            

def create_and_save_images(params):
    iteration, background, progress_queue = params
    positive_frame, negative_frame, black_image = create_image(background)

    cv2.imwrite("./val_frames/" + str(iteration) + '.png', positive_frame)
    cv2.imwrite("./val_mask/" + str(iteration) + '.png', black_image)
    
    iteration += 1
    cv2.imwrite("./val_frames/" + str(iteration) + '.png', negative_frame)
    cv2.imwrite("./val_mask/" + str(iteration) + '.png', black_image)

    progress_queue.put(2)  # Update progress for two images

    return iteration

def main():
    iteration = 0
    progress = 0
    
    for _, _, bgs in os.walk("./background"):
        for bg in bgs:
            background = cv2.imread('./background/' + bg)

            params_list = [(i, background, mp.Queue()) for i in range(2)]
            pool = mp.Pool(mp.cpu_count())
            manager = mp.Manager()

            progress_queue = manager.Queue()

            for i in range(2):
                    params_list[i] = (iteration + 2*i, background, progress_queue)

            pool.map(create_and_save_images, params_list)

            pool.close()
            pool.join()

                        # Print progress
            while not progress_queue.empty():
                progress += progress_queue.get()
                print(progress)


            iteration += 4
           # for _ in range(50):
            #    positive_frame, negative_frame, black_image =  create_image(background)
#

             #   cv2.imwrite("./val_frames/" + str(iteration) + '.png', positive_frame)
             #   cv2.imwrite("./val_mask/"  +str(iteration) + '.png', black_image)

              #  iteration += 1
             #   cv2.imwrite("./val_frames/" + str(iteration) + '.png', negative_frame)
              #  cv2.imwrite("./val_mask/" + str(iteration) + '.png', black_image)
             #   iteration += 1

              #  progress += 1
             #   print(progress)


if __name__ == "__main__":
    main()