"""utils for images"""

# System imports

# 3rd party imports
import cv2
import dlib
import numpy as np


# local imports

def get_random_face_from_img(image):
    """
    returns the bounding box for first face from an image
    """
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    if dets:
        return dets[0]
    else:
        return False


def crop_img(image, drect):
    """
    crop the image with a drect
    """
    # make rect from drect
    rect = dlib.rectangle(drect)
    # print("ROI shape: {}\nlipRect: {}".format(image.shape, rect))
    # crop with max(width, heigh) and min(0)
    max_x = image.shape[1] - 1
    max_y = image.shape[0] - 1
    return image[np.clip(rect.top(), 0, max_y):np.clip(rect.bottom(), 0, max_y),
                 np.clip(rect.left(), 0, max_x):np.clip(rect.right(), 0, max_x)]


def to_img_space(roi_rect, rect):
    """
    returns a rectangle in the coordinate space of the whole image

    :param roi_rect: rectangle in Image Space, which contains the rect
    :type roi_rect: dlib.drectangle
    :param rect: rectangle in ROI Space
    :type rect: dlib.drectangle
    """
    return dlib.drectangle(roi_rect.left() + rect.left(), roi_rect.top() + rect.top(),
                           roi_rect.left() + rect.right(),
                           roi_rect.top() + rect.bottom())


def resize_and_zero_padding(image, shape):
    """
    Resizes a given image to the disered shape without changing the
    (width to height)ratio.
    (On pixel base its not suitable to resize to the exact same ratio - closest with
    rounding is applied)
    Applies zeroPadding to ensure the desired size.
    :param image: The image for resizing
    :type image: numpyarray from openCV
    :param shape: the desired shape(width, height)
    :type shape: tuple of ints
    :returns: resized image
    """
    (h, w) = image.shape[:2]
    width_faktor = shape[0] / w
    height_faktor = shape[1] / h
    faktor = min([width_faktor, height_faktor])
    # print("Faktor: {}".format(faktor))
    multiplied_size = (int(round(w * faktor)), int(round(h * faktor)))
    # print ("Multpilied size: {}".format(multipliedSize))

    image = cv2.resize(image, multiplied_size)
    # cant be exactly the same ratio because resize can only resize to full pixels -
    # take big images(LAW of big numbers
    # will take care)
    width_delta = abs(image.shape[1] - shape[0])
    height_delta = abs(image.shape[0] - shape[1])
    image = cv2.copyMakeBorder(image, 0, height_delta, 0, width_delta,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # image = np.pad(image, [(0, heightDelta), (0, widthDelta)], 'constant')
    assert min([height_delta,
                width_delta]) == 0, \
        "Padding must only be aplied on one side. widthDelta: {}, " \
        "heightDelta: {}".format(
        width_delta, height_delta)
    # ToDo: needed when in test case?
    assert image.shape[0] == shape[1], \
        "Desired height was {} transformed to {}".format(shape[1], image.shape[0])
    assert image.shape[1] == shape[0], \
        "Desired width was {} transformed to {}".format(shape[0], image.shape[1])

    return image


def convert_sample_to_video(data, path, fps=25, codec='MP4V'):
    """
    converts a sample(data) to a video stream.

    :param data: numpy array of shape (timesteps, *img.shape)
    :type data: numpy array
    :param path: path to write the video to 
    :type path: String
    :param fps: Frames per second
    :type fps: integer
    :param codec: Video codec
    :type codec: String
    """
    size = (data.shape[1], data.shape[2])
    out = cv2.VideoWriter()
    out.open(path, cv2.VideoWriter_fourcc(*codec), fps, size, True)
    for img in data:
        # convert to BGR
        out.write(img)
    out.release()
