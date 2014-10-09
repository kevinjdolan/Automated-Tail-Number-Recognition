import time
import logging

from atnr import constants

class Profile(object):

    def __init__(self, label="Profile"):
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, extype, exvalue, traceback):
        duration = time.time() - self.start
        logging.debug("%s: %ss" % (self.label, duration))

def isReasonableSegment(minX, maxX, minY, maxY, width, height):
    # TODO: what is the best place for this?

    subheight = float(1+maxY-minY)
    subwidth = float(1+maxX-minX)

    xlow = (subwidth <= max(8, 0.01 * width))
    ylow = (subheight <= max(8, 0.01 * height))
    xhigh = (subwidth > width/4)
    yhigh = (subheight > height/2)

    aspectRatio = subwidth / subheight
    tooWide = aspectRatio > (constants.LETTER_ASPECT_RATIO_MEAN +  3 * constants.LETTER_ASPECT_RATIO_STD)

    return not (xlow or xhigh or ylow or yhigh or tooWide)