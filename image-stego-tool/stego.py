#import libraries
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def mse(imageA, imageB):

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    import cv2
    imageA = cv2.imread(imageA)
    imageB = cv2.imread(imageB)
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f" % m)
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()

def subtract(src,dest):
    import cv2
    img = cv2.imread(src)
    img2 = cv2.imread(dest)
    print("IMG1")
    print(img[0])
    print("IMG2")
    print(img2[0])
    cv2.imshow('image', img2-img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def visualizeLastBitGrayscale(src,dest):
    img = Image.open(src, 'r')
    width, height = img.size
    array = np.array(list(img.getdata()))
    print('Shape: ',array.shape)
    if img.mode == 'RGB':
        n = 3
    elif img.mode == 'RGBA':
        n = 4

    total_pixels = array.size // n

    index = 0
    for p in range(total_pixels-1):
        for q in range(0, 3):
            bits = bin(array[p][q])[2:10]
            while (len(bits) < 8):
                bits = '0' + bits

            if(bits[7] == '0'):
                array[p][0] = 255
                array[p][1] = 255
                array[p][2] = 255
                q = 4
            else:
                array[p][0] = 0
                array[p][1] = 0
                array[p][2] = 0
                q = 4
            index += 1

    array = array.reshape(height, width, n)
    enc_img = Image.fromarray(array.astype('uint8'), img.mode)
    enc_img.save(dest)

    import cv2
    img = cv2.imread(dest)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualizeLastBit(src,dest):
    img = Image.open(src, 'r')
    width, height = img.size
    array = np.array(list(img.getdata()))
    print('Shape: ',array.shape)
    if img.mode == 'RGB':
        n = 3
    elif img.mode == 'RGBA':
        n = 4

    total_pixels = array.size // n

    index = 0
    for p in range(total_pixels-1):
        for q in range(0, 3):
            bits = bin(array[p][q])[2:10]
            while (len(bits) < 8):
                bits = '0' + bits

            if(bits[7] == '0'):
                array[p][q] = 255
            else:
                array[p][q] = 0
            index += 1

    array = array.reshape(height, width, n)
    enc_img = Image.fromarray(array.astype('uint8'), img.mode)
    enc_img.save(dest)

    import cv2
    img = cv2.imread(dest)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#encoding function
def Encode(src, message, dest):

    img = Image.open(src, 'r')
    width, height = img.size
    array = np.array(list(img.getdata()))

    if img.mode == 'RGB':
        n = 3
    elif img.mode == 'RGBA':
        n = 4

    total_pixels = array.size//n
    if(not message):
        with open('story.txt', 'r') as file:
            message = file.read().replace('\n', '')
    print(message)
    message += "$t3g0"
    b_message = ''.join([format(ord(i), "08b") for i in message])
    req_pixels = len(b_message)

    if req_pixels > total_pixels:
        print("ERROR: Need larger file size")

    else:
        index=0
        for p in range(total_pixels):
            for q in range(0, 3):
                if index < req_pixels:
                    bits = bin(array[p][q])[2:10]
                    while (len(bits) < 8):
                        bits = '0' + bits
                    #print('Before: ',int(bits, 2),'/',bits,' after: ',int(bits[0:7] + b_message[index], 2),'/',bits[0:7] + b_message[index],' message: ',b_message[index])

                    array[p][q] = int(bits[0:7] + b_message[index], 2)
                    index += 1

        array=array.reshape(height, width, n)
        enc_img = Image.fromarray(array.astype('uint8'), img.mode)
        enc_img.save(dest)
        print("Image Encoded Successfully")


#decoding function
def Decode(src):

    img = Image.open(src, 'r')
    array = np.array(list(img.getdata()))

    if img.mode == 'RGB':
        n = 3
    elif img.mode == 'RGBA':
        n = 4

    total_pixels = array.size//n

    hidden_bits = ""
    for p in range(total_pixels):
        for q in range(0, 3):
            hidden_bits += (bin(array[p][q])[2:][-1])

    hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]

    message = ""
    for i in range(len(hidden_bits)):
        if message[-5:] == "$t3g0":
            break
        else:
            message += chr(int(hidden_bits[i], 2))
    if "$t3g0" in message:
        print("Hidden Message:", message[:-5])
    else:
        print("No Hidden Message Found")

#main function
def Stego():
    print("--Welcome to $t3g0--")
    print("1: Encode")
    print("2: Decode")
    print("3: Compare")
    print("4: Subtract")
    print("5: mse")
    func = input()

    if func == '1':
        print("Enter Source Image Path")
        src = input()
        print("Enter Message to Hide")
        message = input()
        print("Enter Destination Image Path")
        dest = input()
        print("Encoding...")
        Encode(src, message, dest)

    elif func == '2':
        print("Enter Source Image Path")
        src = input()
        print("Decoding...")
        Decode(src)
    elif func == '3':
        print("Enter Source Image Path")
        src = input()
        print("Enter Destination Image Path")
        dest = input()
        print("Visualizing...")
        visualizeLastBitGrayscale(src, dest)
    elif func == '4':
        print("Enter Source Image Path")
        src = input()
        print("Enter Destination Image Path")
        dest = input()
        print("Visualizing...")
        subtract(src, dest)
    elif func == '5':
        print("Enter Source Image Path")
        src = input()
        print("Enter Destination Image Path")
        dest = input()
        print("MSE...")
        compare_images(src, dest,"MSE")
    else:
        print("ERROR: Invalid option chosen")

Stego()
