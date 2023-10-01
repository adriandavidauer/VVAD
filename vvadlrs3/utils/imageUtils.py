"""utils for images"""

# System imports

# 3rd party imports
import cv2
import dlib
import numpy as np


# local imports

def getRandomFaceFromImage(image):
    """ returns the bounding box for first face from an image

    Args:
        image (image): image in R, G, B

    """
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    if dets:
        return dets[0]
    else:
        return False


def crop_img(image, drect):
    """ Crop the provided image with a drect

    Args:
        image (image): Image file as....
        drect (drect): ??
    """
    rect = dlib.rectangle(drect)
    max_x = image.shape[1] - 1
    max_y = image.shape[0] - 1
    return image[np.clip(rect.top(), 0, max_y):np.clip(rect.bottom(), 0, max_y),
                 np.clip(rect.left(), 0, max_x):np.clip(rect.right(), 0, max_x)]


def to_img_space(roi_rect, rect):
    """ returns a rectangle in the coordinate space of the whole image

    Args:
        ROIrect (dlib.drectangle): rectangle in Image Space which contains the rect.
        rect (dlib.drectangle): rectangle in ROI Space

    Returns:
        Rectangle in coordinate space of the image
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


    Args:
        image (numpy array from openCV): The image for resizing
        shape (tuple of ints): the desired shape in width and height

    Returns:
        image (numpy array from openCV): resized image

    """
    (h, w) = image.shape[:2]
    widthFaktor = shape[0] / w
    heightFaktor = shape[1] / h
    faktor = min([widthFaktor, heightFaktor])
    multipliedSize = (int(round(w * faktor)), int(round(h * faktor)))

    image = cv2.resize(image, multipliedSize)
    # cant be exactly the same ratio because resize can only resize to full pixels -
    # take big images(LAW of big numbers will take care)

    widthDelta = abs(image.shape[1] - shape[0])
    heightDelta = abs(image.shape[0] - shape[1])
    image = cv2.copyMakeBorder(image, 0, heightDelta, 0, widthDelta,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    assert min([heightDelta,
                widthDelta]) == 0, "Padding must only be aplied on one side. " \
                                   "widthDelta: {}, heightDelta: {}".format(
        widthDelta, heightDelta)
    assert image.shape[0] == shape[1], "Desired height was {} transformed to {}".format(
        shape[1], image.shape[0])
    assert image.shape[1] == shape[0], "Desired width was {} transformed to {}".format(
        shape[0], image.shape[1])

    return image


def convert_sample_to_video(data, path, fps=25, codec='MP4V'):
    """ converts a sample(data) to a video stream.

    Args:
        data (numpy array): numpy array of shape (timesteps, *img.shape)
        path (str): Path to save the video to
        fps (int): Frames per second of data (default: 25)
        codec (str): Video-codec of data (default: 'MP4V')
    """
    size = (data.shape[1], data.shape[2])
    out = cv2.VideoWriter()
    out.open(path, cv2.VideoWriter_fourcc(*codec), fps, size, True)
    for img in data:
        out.write(img)
    out.release()
