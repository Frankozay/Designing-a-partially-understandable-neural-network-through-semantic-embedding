import numpy as np
#from scipy.ndimage import rotate, interpolation, gaussian_filter, uniform_filter
#from scipy.misc import imresize
#import matplotlib.pyplot as plt


# NOTE
# Images are assumed to be uint8 0-255 valued.
# For augment function:
#   images shape: (batch_size, height, width, channels=3)
#   labels shape: (batch_size, 3)
#
#def addBlotch(image, max_dims=[0.2, 0.2]):
#    # add's small black/white box randomly in periphery of image
#    new_image = np.copy(image)
#    shape = new_image.shape
#    max_x = shape[0] * max_dims[0]
#    max_y = shape[1] * max_dims[1]
#    rand_x = 0
#    rand_y = np.random.randint(low=0, high=shape[1])
#    rand_bool = np.random.randint(0, 2)
#    if rand_bool == 0:
#        rand_x = np.random.randint(low=0, high=max_x)
#    else:
#        rand_x = np.random.randint(low=(shape[0] - max_x), high=shape[0])
#    size = np.random.randint(low=1, high=7)  # size of each side of box
#    new_image[rand_x:(size + rand_x), rand_y:(size + rand_y), :] = np.random.randint(0, 256)
#    return new_image
#
#
#def shift(image, max_amt=0.2):
#    new_img = np.copy(image)
#    shape = new_img.shape
#    max_x = int(shape[0] * max_amt)
#    max_y = int(shape[1] * max_amt)
#    x = np.random.randint(low=-max_x, high=max_x)
#    y = np.random.randint(low=-max_y, high=max_y)
#    return interpolation.shift(new_img, shift=[x, y, 0])
#
#
#def addNoise(image, amt=0.005):
#    row, col, ch = image.shape
#    mean = 0
#    var = 0.1
#    sigma = var ** 0.5
#    gauss = np.random.normal(mean, sigma, (row, col, ch))
#    gauss = gauss.reshape(row, col, ch)
#    noisy_img = image + gauss
#    return np.array(noisy_img)
#
#
#def img_rotate(image):
#    randnum = np.random.randint(1, 360)
#    new_image = np.copy(image)
#    return rotate(new_image, angle=randnum, reshape=False)


def img_flip_lr(image):
    return np.copy(image[:, -1::-1, :])


def img_flip_ud(image):
    return np.copy(image[-1::-1, :, :])


#def img_blur(image):
#    randnum = np.random.randint(1, 6)
#    new_image = np.copy(image)
#    new_image = gaussian_filter(new_image, sigma=randnum)
#    new_image = uniform_filter(new_image, size=randnum)
#    return new_image


def img_padding_crop(image):
    lx, ly, lz = image.shape
    padding_size = 4
    new_img = np.zeros([lx + padding_size * 2, ly + padding_size * 2, lz])
    new_img[padding_size:lx + padding_size, padding_size:ly + padding_size, :] = image
    x1 = np.random.randint(0, padding_size * 2)
    y1 = np.random.randint(0, padding_size * 2)
    return new_img[x1:x1 + lx, y1:y1 + ly, :]


#def img_crop(image):
#    lx, ly, lz = image.shape
#    x = np.random.randint(low=0, high=lx - int(lx * 3 / 4))
#    y = np.random.randint(low=0, high=ly - int(ly * 3 / 4))
#    #    x = 4
#    #    y = 4
#    new_image = np.copy(image)
#    crop_img = new_image[x:x + int(lx * 3 / 4), y:y + int(ly * 3 / 4), :]
#    return imresize(crop_img, [lx, ly])


def show_aug_results(images):
    Num = images.shape[0]
    if Num > 9:
        Num = 9
    NumRows = 3
    NumCols = Num / NumRows + Num % NumRows

    plt.figure()
    for i in range(Num):
        plt.subplot(NumRows, NumCols, i + 1)
        plt.imshow(images[i])

    plt.savefig('Data_Augmentation.png')
    plt.close()


# randomly manipulates image
# rotate, flip along axis, add blotch, shift
def data_augment(imgs):
    shape = imgs.shape
    # new_imgs = np.zeros((shape[0], shape[1], shape[2], shape[3]))
    new_imgs = np.copy(imgs)
    for i in range(shape[0]):
        cur_img = np.copy(imgs[i])
        new_imgs[i] = img_padding_crop(cur_img)
        if np.random.random() > 0.5:
            cur_img = np.copy(new_imgs[i])
            new_imgs[i] = img_flip_lr(cur_img)

    return new_imgs


def augment(images, labels=None, amplify=2):
    # INPUT:
    # images shape: (batch_size, height, width, channels=3)
    # labels shape: (batch_size, 3)
    ops = {
        0: addBlotch,
        1: shift,
        2: addNoise,
        3: img_rotate,
        4: img_flip_lr,
        5: img_flip_ud,
        6: img_blur,
        7: img_crop
    }

    shape = images.shape
    new_images = np.zeros(((amplify * shape[0]), shape[1], shape[2], shape[3]))
    if labels is not None:
        new_labels = np.zeros(((amplify * shape[0])))
    for i in range(images.shape[0]):
        cur_img = np.copy(images[i])
        new_images[i] = cur_img
        if labels is not None:
            new_labels[i] = np.copy(labels[i])
        for j in range(1, amplify):
            add_r = (j * shape[0])
            which_op = np.random.randint(low=0, high=8)
            dup_img = np.zeros((1, shape[1], shape[2], shape[3]))
            new_images[i + add_r] = ops[which_op](cur_img)
            if labels is not None:
                new_labels[i + add_r] = np.copy(labels[i])
    if labels is not None:
        return new_images.astype(np.uint8), new_labels.astype(np.uint8)
    else:
        return new_images.astype(np.uint8)
