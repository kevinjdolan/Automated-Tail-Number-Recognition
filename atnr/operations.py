import math
import numpy
import logging
import scipy.stats

from atnr import constants
from atnr.util import Profile

from skimage import filter
from skimage import measure
from skimage import transform
from skimage import morphology

# abstractions

class BaseOperation(object):

    __abstract__ = True
    __inputs__ = None
    __outputs__ = None

    @classmethod
    def execute(cls, **environment):
        operation = cls()
        return operation._execute(**environment)

    def __init__(self, args=None):
        self.args = args or {}
        self.execute = self._execute

    def operate(self, environment):
        with Profile(self.__class__.__name__):
            inputEnvironment = self._getInputEnvironment(environment)
            outputEnvironment = self.run(**inputEnvironment)
            self._setOutputEnvironment(environment, outputEnvironment)

    def run(self, **kwargs):
        raise NotImplemented

    def _execute(self, **environment):
        self.operate(environment)
        return environment[self.__outputs__[0]]

    def _getInputEnvironment(self, environment):
        inputEnvironment = {}
        for localKey in self.__inputs__:
            environmentKey = self.args.get(localKey, localKey)
            inputEnvironment[localKey] = environment[environmentKey]
        return inputEnvironment

    def _setOutputEnvironment(self, environment, outputEnvironment):
        for localKey in outputEnvironment.keys():
            environmentKey = self.args.get(localKey, localKey)
            environment[environmentKey] = outputEnvironment[localKey]

class BaseThreshold(BaseOperation):

    __abstract__ = True

    __inputs__ = ['image']
    __outputs__ = ['binaryImage']

class BaseCandidateExtractor(BaseOperation):

    __abstract__ = True

    __inputs__ = ['labeledImage']
    __outputs__ = ['candidates']

    def run(self, labeledImage):
        height = labeledImage.shape[0]
        width = labeledImage.shape[1]

        labelBboxes = FindLabelBboxes.execute(labeledImage=labeledImage)

        candidates = []
        for (label, (minX,maxX,minY,maxY)) in labelBboxes.items():
            if util.isReasonableSegment(minX, maxX, minY, maxY, width, height):
                extracted = ExtractSegment.execute(
                    image=labeledImage, 
                    label=label, 
                    bbox=(minX,maxX,minY,maxY),
                )
                probability = self.evaluateSegment(extracted, label)
                if probability:
                    candidates.append({
                        'label': label, 
                        'probability': probability,
                        'bbox': (minX, maxX, minY, maxY),
                        'width': maxX-minX,
                        'height': maxY-minY,
                        'xloc': (maxX+minX)/2./width,
                        'yloc': (maxY+minY)/2./height,
                        'xsize': (maxX-minX)/float(width),
                        'ysize': (maxY-minY)/float(height),
                        'image': extracted,
                    })

        candidates = sorted(candidates, key=lambda x: x['probability'])

        return {'candidates': candidates}

class BaseSegmenter(BaseOperation):

    __abstract__ = True

    __inputs__ = ['image']
    __outputs__ = ['labeledImage']

class ComposeOperations(BaseOperation):

    suboperations = None

    @classmethod
    def compose(cls, environment, suboperations):
        operation = ComposeOperations(
            suboperations=suboperations,
        )
        operation.operate(environment)
        return environment

    def __init__(self, suboperations=None, **kwargs):
        BaseOperation.__init__(self, **kwargs)
        self.suboperations = suboperations
        self.__inputs__ = self.suboperations[0].__inputs__
        self.__outputs__ = []
        for suboperation in self.suboperations:
            self.__outputs__ += suboperation.__outputs__

    def operate(self, environment):
        localEnvironment = self._getInputEnvironment(environment)
        for suboperation in self.suboperations:
            suboperation.operate(localEnvironment)
        self._setOutputEnvironment(environment, localEnvironment)

# implementations

class AdaptiveThreshold(BaseThreshold):

    def run(self, image):
        fieldSize = image.shape[0] * constants.TAIL_AREA_BBOX_HEIGHT_MEAN
        thresholded = filter.threshold_adaptive(image, fieldSize, method='gaussian')
        return {'binaryImage': thresholded}

class Clip(BaseOperation):

    __inputs__ = ['image']
    __outputs__ = ['clippedImage']

    def run(self, image):
        (minX,maxX,minY,maxY) = self.findBBox(image)
        return image[minY:maxY,minX:maxX]

    def findBbox(self, image):
        minX = image.shape[1]
        maxX = 0
        minY = image.shape[0]
        maxY = 0
        iter = numpy.nditer(image, flags=['multi_index'])
        while not iter.finished:
            value = numpy.float_(iter[0])
            if value:
                y, x = iter.multi_index
                if x < minX:
                    minX = x
                if x > maxX:
                    maxX = x
                if y < minY:
                    minY = y
                if y > maxY:
                    maxY = y
            
            iter.iternext()
        return minX,maxX,minY,maxY

class ComputeNeighborhoodImage(BaseOperation):

    __inputs__ = ['image', 'neighborhoodMask', 'neighborhoodIntensityProfile']
    __outputs__ = ['neighborhoodImage']

    def run(self, image, neighborhoodMask, neighborhoodIntensityProfile):
        bbox = neighborhoodMask['bbox']
        neighborhoodImageRaw = image[bbox[2]:bbox[3],bbox[0]:bbox[1]]

        (mean, std) = neighborhoodIntensityProfile
        neighborhoodImage = RestrictImageIntensityRange.execute(
            image=neighborhoodImageRaw,
            intensityMean=mean,
            intensityStd=std,
        )
        
        return {'neighborhoodImage': neighborhoodImage}

class CorrectOrientation(BaseOperation):

    __inputs__ = ['image', 'transformation']
    __outputs__ = ['correctedImage']

    def run(self, image, transformation):
        transformer = transform.ProjectiveTransform()
        transformer.estimate(
            numpy.array(transformation['to']), 
            numpy.array(transformation['from']),
        )
        warped = transform.warp(
            image, 
            transformer, 
            output_shape=transformation['shape'],
        )
        warped = numpy.array(warped.tolist()) # why is this necessary?
        return {'correctedImage': warped}

class EvaluateNeighborhoodIntensityProfile(BaseOperation):

    __inputs__ = ['image', 'neighborhood']
    __outputs__ = ['neighborhoodIntensityProfile']

    def run(self, image, neighborhood):
        intensities = []
        for member in neighborhood:
            minX = member['bbox'][0]
            minY = member['bbox'][2]
            iter = numpy.nditer(member['image'], flags=['multi_index'])
            while not iter.finished:
                if numpy.bool_(iter[0]):
                    y, x = iter.multi_index
                    intensities.append(image[y+minY,x+minX])
                iter.iternext()
        return {
            'neighborhoodIntensityProfile': (
                numpy.mean(intensities),
                numpy.std(intensities),
            )
        }

class ExtractSegment(BaseOperation):

    __inputs__ = ['image', 'label', 'bbox']
    __outputs__ = ['extractedSegment']

    def run(self, image, label, bbox):
        if bbox:
            image = image[bbox[2]:bbox[3],bbox[0]:bbox[1]]

        def xExtract(x):
            if x == label:
                return 1.0
            else:
                return 0.0
        vExtract = numpy.vectorize(xExtract)
        
        return {'extractedSegment': vExtract(image)}

class FindLabelBboxes(BaseOperation):

    __inputs__ = ['labeledImage']
    __outputs__ = ['labelBboxes']

    def run(self, labeledImage):
        labelBboxes = {}
        iter = numpy.nditer(labeledImage, flags=['multi_index'])
        while not iter.finished:
            label = numpy.int_(iter[0])
            if label not in labelBboxes:
                labelBboxes[label] = [
                    iter.multi_index[1], # minX
                    iter.multi_index[1], # maxX
                    iter.multi_index[0], # minY
                    iter.multi_index[0], # maxY
                ]
            else:
                labelBbox = labelBboxes[label]
                y, x = iter.multi_index
                if x < labelBbox[0]:
                    labelBbox[0] = x
                if x > labelBbox[1]:
                    labelBbox[1] = x
                if y < labelBbox[2]:
                    labelBbox[2] = y
                if y > labelBbox[3]:
                    labelBbox[3] = y
            
            iter.iternext()
        return {'labelBboxes': labelBboxes}

class FindNeighborhoodMask(BaseOperation):

    __inputs__ = ['neighborhood']
    __outputs__ = ['neighborhoodMask', 'neighborhoodTransformation']

    def run(self, neighborhood):
        neighborhood = sorted(neighborhood, key=lambda x: x['xloc'])
        indexes, gaps = self.findGaps(neighborhood)

        numberInterior = len(indexes) + gaps
        if numberInterior < constants.TAIL_NUMBER_LENGTH:
            leftBbox, rightBbox = self.expandNeighborhood(
                neighborhood, indexes, numberInterior
            )
        else:
            leftBbox = neighborhood[0]['bbox']
            rightBbox = neighborhood[-1]['bbox']

        leftBbox = self.growBbox(leftBbox)
        rightBbox = self.growBbox(rightBbox)


        bbox = (
            leftBbox[0],
            rightBbox[1],
            min(leftBbox[2], rightBbox[2]),
            max(leftBbox[3], rightBbox[3]),
        )

        leftTop = (leftBbox[0]-bbox[0],leftBbox[2]-bbox[2]) #(leftBbox[2]-bbox[2], leftBbox[0]-bbox[0])
        leftBottom = (leftBbox[0]-bbox[0],leftBbox[3]-bbox[2])#(leftBbox[3]-bbox[2], leftBbox[0]-bbox[0])
        rightTop = (rightBbox[1]-bbox[0],rightBbox[2]-bbox[2])#(rightBbox[2]-bbox[2], rightBbox[1]-bbox[0])
        rightBottom = (rightBbox[1]-bbox[0],rightBbox[3]-bbox[2])#(rightBbox[3]-bbox[2], rightBbox[1]-bbox[0])

        meanHeight = ((leftBbox[3]-leftBbox[2])+(rightBbox[3]-rightBbox[2]))/2
        width = rightBbox[1]-leftBbox[0]
        transformation = {
            'from': [leftTop, rightTop, rightBottom, leftBottom],
            'to': [(0,0),(width,0),(width,meanHeight),(0,meanHeight)],
            'shape': (meanHeight, width),
        }
        
        return {
            'neighborhoodMask': {'bbox': bbox,},
            'neighborhoodTransformation': transformation,
        }

    # def computeMask(self, leftBbox, rightBbox, bbox):
    #     bboxWidth = bbox[1] - bbox[0]
    #     bboxHeight = bbox[3] - bbox[2]
    #     mask = numpy.zeros((bboxHeight+1,bboxWidth+1))
    #     for box in [leftBbox, rightBbox]:
    #         for y in [box[2]-bbox[2], box[3]-bbox[2]]:
    #             for x in [box[0]-bbox[0], box[1]-bbox[0]]:
    #                 mask[y,x] = 1
    #     return morphology.convex_hull_image(mask)

    def expandNeighborhood(self, neighborhood, indexes, numberInterior):
        X = indexes
        minXPredictor = self.linearRegressionFunction(
            X, [member['bbox'][0] for member in neighborhood]
        )
        maxXPredictor = self.linearRegressionFunction(
            X, [member['bbox'][1] for member in neighborhood]
        )
        minYPredictor = self.linearRegressionFunction(
            X, [member['bbox'][2] for member in neighborhood]
        )
        maxYPredictor = self.linearRegressionFunction(
            X, [member['bbox'][3] for member in neighborhood]
        )

        exteriorNeeded = constants.TAIL_NUMBER_LENGTH - numberInterior

        leftIndex = -exteriorNeeded 
        rightIndex = indexes[-1] + exteriorNeeded

        leftBbox = (minXPredictor(leftIndex), maxXPredictor(leftIndex), minYPredictor(leftIndex), maxYPredictor(leftIndex))
        rightBbox = (minXPredictor(rightIndex), maxXPredictor(rightIndex), minYPredictor(rightIndex), maxYPredictor(rightIndex))

        return leftBbox, rightBbox

    def findGaps(self, neighborhood):
        meanWidth = numpy.mean([member['width'] for member in neighborhood])
        
        gaps = 0
        indexes = [0]
        index = 1
        lastMaxX = neighborhood[0]['bbox'][1]
        for member in neighborhood[1:]:
            overlap = (lastMaxX - member['bbox'][0])
            # negative overlap indicates a gap -- is it tolerable?
            tolerableOverlap = (constants.LETTER_NEIGHBORHOOD_OVERLAP_VARIANCE_STATS[0] - 2 * constants.LETTER_NEIGHBORHOOD_OVERLAP_VARIANCE_STATS[1])
            if overlap/meanWidth < tolerableOverlap:
                space = -overlap
                idealOverlap = constants.LETTER_NEIGHBORHOOD_OVERLAP_VARIANCE_STATS[0] * meanWidth
                number = max(1, int(math.floor((space+idealOverlap)/(meanWidth-idealOverlap))))
                gaps += number
                index += number

            indexes.append(index)
            index += 1
            lastMaxX = member['bbox'][1]

        return indexes, gaps

    def growBbox(self, bbox):
        (minX, maxX, minY, maxY) = bbox

        xratio = constants.LETTER_NEIGHBORHOOD_X_SIZE_RELATIVE_VARIANCE_STATS[0]/3
        yratio = constants.LETTER_NEIGHBORHOOD_Y_SIZE_RELATIVE_VARIANCE_STATS[0]/3

        width = maxX-minX
        height = maxY-minY

        return (
            int(minX - xratio * width), 
            int(maxX + xratio * width), 
            int(minY - yratio * height), 
            int(maxY + yratio * height),
        )

    def linearRegressionFunction(self, X, Y):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X,Y)
        def func(x):
            return slope * x + intercept
        return func

class FindSegmentNeighborhood(BaseOperation):

    __inputs__ = ['candidates']
    __outputs__ = ['neighborhood']

    def run(self, candidates):
        neighborhoods = []
        for candidate in candidates:
            matched = False
            for neighborhood in neighborhoods:
                added = self.possiblyAddToNeighborhood(candidate, neighborhood)
                if added:
                    matched = True
                    break
            if not matched:
                neighborhoods.append([candidate])

        neighborhoods = [neighborhood for neighborhood in neighborhoods if len(neighborhood) > 1]
        neighborhoods = sorted(neighborhoods, key=lambda x: len(x), reverse=True)
        return {'neighborhood': neighborhoods[0]}

    def possiblyAddToNeighborhood(self, candidate, neighborhood):
        for member in list(neighborhood):
            drxloc = abs(member['xloc']-candidate['xloc'])/min(member['xsize'],candidate['xsize'])
            dryloc = abs(member['yloc']-candidate['yloc'])/min(member['ysize'],candidate['ysize'])
            drxsize = abs(member['xsize']-candidate['xsize'])/min(member['xsize'],candidate['xsize'])
            drysize = abs(member['ysize']-candidate['ysize'])/min(member['ysize'],candidate['ysize'])

            withinXLoc = drxloc < constants.LETTER_NEIGHBORHOOD_X_LOC_RELATIVE_VARIANCE_STATS[0] + 3 * constants.LETTER_NEIGHBORHOOD_X_LOC_RELATIVE_VARIANCE_STATS[1]
            withinYLoc = dryloc < constants.LETTER_NEIGHBORHOOD_Y_LOC_RELATIVE_VARIANCE_STATS[0] + 3 * constants.LETTER_NEIGHBORHOOD_Y_LOC_RELATIVE_VARIANCE_STATS[1]
            withinXSize = drxsize < constants.LETTER_NEIGHBORHOOD_X_SIZE_RELATIVE_VARIANCE_STATS[0] + 3 * constants.LETTER_NEIGHBORHOOD_X_SIZE_RELATIVE_VARIANCE_STATS[1]
            withinYSize = drysize < constants.LETTER_NEIGHBORHOOD_Y_SIZE_RELATIVE_VARIANCE_STATS[0] + 3 * constants.LETTER_NEIGHBORHOOD_Y_SIZE_RELATIVE_VARIANCE_STATS[1]

            if withinXLoc and withinYLoc and withinXSize and withinYSize:
                neighborhood.append(candidate)
                return True

class GlobalMeanThreshold(BaseThreshold):

    def run(self, image):
        mean = numpy.mean(image)
        def xBisect(x):
            if x > mean:
                return 1
            else:
                return 0
        vBisect = numpy.vectorize(xBisect)
        thresholded = vBisect(image)
        return {'binaryImage': thresholded}

class KnnPixelVectorCandidateExtractor(BaseCandidateExtractor):

    def evaluateSegment(self, segment, label):
        edge = knnPixel.KNN_PIXEL_CLASSIFIER_SIZE
        vector = Squarify(edge=edge).execute(image=segment, edge=edge).ravel()
        vector = vector.astype(bool)
        probability = knnPixel.KNN_PIXEL_CLASSIFIER.predict_proba(vector)[0][1]
        if probability > 0.3:
            return probability
        else:
            return 0.0

class LabelConnectedComponents(BaseOperation):

    __inputs__ = ['binaryImage']
    __outputs__ = ['labeledImage']

    def __init__(self, ignoreZero=False, **kwargs):
        BaseOperation.__init__(self, **kwargs)
        self.ignoreZero = ignoreZero

    def run(self, binaryImage):
        if self.ignoreZero:
            background = 0
        else:
            background = None
        return {'labeledImage': measure.label(binaryImage, background=background)}

class RestrictImageIntensityRange(BaseOperation):

    __inputs__ = ['image', 'intensityMean', 'intensityStd']
    __outputs__ = ['restrictedImage']

    def run(self, image, intensityMean, intensityStd):
        intensityNorm = scipy.stats.norm(intensityMean, intensityStd)
        adjust = 1. / intensityNorm.pdf(intensityMean)

        def xRestrict(x):
            return adjust * intensityNorm.pdf(x)
        vRestrict = numpy.vectorize(xRestrict)
            
        return {'restrictedImage': vRestrict(image)}

class Squarify(BaseOperation):

    __inputs__ = ['image']
    __outputs__ = ['normalizedImage']

    def __init__(self, edge, **kwargs):
        BaseOperation.__init__(self, **kwargs)
        self.edge = edge

    def run(self, image):
        bigger = max(image.shape[0], image.shape[1])
        canvas = numpy.zeros((bigger,bigger))
        canvas[0:image.shape[0],0:image.shape[1]] = image
        scaled = transform.resize(canvas, (self.edge, self.edge), order=0)
        return {'normalizedImage': scaled}

class ThresholdSegmenter(BaseSegmenter):

    def __init__(self, threshold, ignoreZero=False, **kwargs):
        BaseSegmenter.__init__(self, **kwargs)
        self.threshold = threshold
        self.ignoreZero = ignoreZero

    def run(self, image):
        return ComposeOperations.compose(
            environment={'image': image},
            suboperations=[
                self.threshold,
                LabelConnectedComponents(self.ignoreZero),
            ]
        )

class TrimVertical(BaseOperation):

    __inputs__ = ['image']
    __outputs__ = ['trimmedImage']

    def run(self, image):
        trimmedImage = numpy.copy(image)

        factor = (constants.LETTER_VERTICAL_HISTOGRAM_MIN_STATS[0] - 1.0 * constants.LETTER_VERTICAL_HISTOGRAM_MIN_STATS[1])
        minSegmentSize = max(2, math.ceil(trimmedImage.shape[0] * factor))

        for x, vector in enumerate(image.T):
            state = False
            counter = 0
            for y, v in enumerate(vector):
                if not state:
                    if v != 0:
                        state = True
                        counter = 1
                else:
                    if v != 0:
                        counter += 1
                    else:
                        state = False
                        if counter < minSegmentSize:
                            trimmedImage[y-counter:y,x:x+1] = 0

        return {'trimmedImage': trimmedImage}