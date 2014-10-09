# contains functinons for accessing the training and testing data

import os
import json

TESTING_DATA_DIR = './data/testing'
TRAINING_DATA_DIR = './data/training'

def getImagePath(id):
    for dir in [TESTING_DATA_DIR, TRAINING_DATA_DIR]:
        if os.path.exists("%s/%s" % (dir, id)):
            return "%s/%s/image.png" % (dir, id)

def getInfo(id):
    for dir in [TESTING_DATA_DIR, TRAINING_DATA_DIR]:
        if os.path.exists("%s/%s" % (dir, id)):
            path = "%s/%s" % (dir, id)
            if os.path.exists("%s/%s/info.json" % (dir, id)):
                with open("%s/%s/info.json" % (dir, id)) as f:
                    obj = json.load(f)
                    obj['id'] = id
                    obj['testing'] = (dir == TESTING_DATA_DIR)
                    obj['path'] = path
                    return obj
            return {
                'id': id, 
                'testing': (dir == TESTING_DATA_DIR),
                'path': path,
            }

def getSegmentations(item, segmentation):
    segmentations = []
    targetPath = "%s/%s" % (item['path'], segmentation)
    if os.path.exists(targetPath):
        for file in os.listdir(targetPath):
            if file.endswith('.png'):
                label = file.replace('.png', '')
                infoPath = "%s/%s.json" % (targetPath, label)
                if os.path.exists(infoPath):
                    with open(infoPath) as f:
                        info = json.load(f)
                        if not info.get('status'):
                            info['status'] = "INFERRED"
                else:
                    info = {'status': "UNMARKED"}
                info['segmentation'] = segmentation
                info['label'] = label
                info['id'] = item['id']
                segmentations.append(info)
    return segmentations

def listData(testing=False):
    dir = _getDir(testing)
    ids = [id for id in os.listdir(dir) if id != '.DS_Store']
    infos = [getInfo(id) for id in ids]
    return infos

def updateInfo(id, info):
    if info['testing']:
        dir = TESTING_DATA_DIR
    else:
        dir = TRAINING_DATA_DIR
    with open("%s/%s/info.json" % (dir, id), 'w') as f:
        json.dump(info, f)

def updateSegmentation(segmentInfo):
    info = getInfo(segmentInfo['id'])
    targetPath = "%s/%s/%s.json" % (
        info['path'], 
        segmentInfo['segmentation'], 
        segmentInfo['label'],
    )
    with open(targetPath, 'w') as f:
        json.dump(segmentInfo, f)

def _getDir(testing):
    if testing:
        return TESTING_DATA_DIR
    else:
        return TRAINING_DATA_DIR