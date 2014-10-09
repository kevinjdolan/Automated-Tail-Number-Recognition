# this script goes through the test and training data, 
# and then segments those images, outputting them for further
# tagging

# it will attempt to autotag any likely candidates based on 
# the tail character location tagger

# warning! this will overwrite any existing data!

import json
import os
import sys

from skimage import io

from atnr import data
from atnr import util
from atnr.operations import *

for testing in [True, False]:
    for item in data.listData(testing):
        if item['tail'] and len(item['tail']) == len(item['characters']):
            print item['id']

            imagePath = data.getImagePath(item['id'])
            image = io.imread(imagePath, True)


            environment = ComposeOperations.compose(
                environment={'image': image},
                suboperations=[
                    ThresholdSegmenter(AdaptiveThreshold()),
                    FindLabelBboxes(),
                ],
            )

            labeledImage = environment['labeledImage']

            lettersByLabel = {}
            for letter, location in zip(item['tail'], item['characters']):
                label = labeledImage[location[1],location[0]]
                lettersByLabel[label] = letter

            targetDir = "%s/candidate-segmentation" % item['path']
            if os.path.exists(targetDir):
                if '--clobber' not in sys.argv:
                    print "WARNING!! OVERWRITING DATA!!"
                    print "If you're sure, run with --clobber"
                    exit()
            else:
                os.mkdir(targetDir)

            height, width = image.shape
            labelBboxes = environment['labelBboxes']
            for (label, (minX,maxX,minY,maxY)) in labelBboxes.items():
                if util.isReasonableSegment(minX, maxX, minY, maxY, width, height):
                    extracted = ExtractSegment.execute(
                        image=labeledImage, 
                        label=label, 
                        bbox=(minX,maxX,minY,maxY),
                    )
                    io.imsave(
                        "%s/%s.png" % (targetDir, label),
                        extracted,
                    )

                    letter = lettersByLabel.get(label)
                    if letter:
                        with open("%s/%s.json" % (targetDir, label), 'w') as f:
                            json.dump({'character': letter}, f)