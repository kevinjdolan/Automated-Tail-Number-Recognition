# a very simple flask app for manually tagging the testing and training data

import flask
import json
import sys
import os

sys.path.append('.')

APP = flask.Flask(__name__)

from atnr import data

@APP.route("/")
def root():
    return flask.render_template('root.jinja2')

@APP.route("/tail-character-locations/")
def tailCharacterLocations():
    pending = getPendingTailCharacterLocations()
    
    id = flask.request.values.get('id')
    if id is None:
        if pending:
            return flask.redirect('/tail-character-locations/?id=%s' % pending[0])
        else:
            return "UR DONE"
    else:
        info = data.getInfo(id)
        charactersJson = json.dumps(info.get('characters', []))
        return flask.render_template('tailCharacterLocations.jinja2',
            info=info,
            charactersJson=charactersJson,
            remaining=len(pending),
        )

@APP.route("/tail-character-locations/", methods=['POST'])
def tailCharacterLocationsSave():
    id = flask.request.values.get('id')
    info = data.getInfo(id)
    info['version'] = 0
    info['tail'] = flask.request.values.get('tail').upper()

    markers = flask.request.values.getlist('markers')
    markers.remove('X')

    characters = []
    for marker in markers:
        delete = flask.request.values.get('delete-' + marker)
        if delete != 'true':
            x = flask.request.values.get('x-' + marker)
            y = flask.request.values.get('y-' + marker)
            if x and y:
                characters.append((int(x),int(y)))
    characters = sorted(characters, key=lambda x: x[0])
    info['characters'] = characters

    data.updateInfo(id, info)

    return flask.redirect('/tail-character-locations/')

def getPendingTailCharacterLocations():
    candidates = []
    for testing in [True, False]:
        for info in data.listData(testing):
            if info.get('version', -1) < 0:
                candidates.append(info['id'])
    return candidates

@APP.route('/segmentation-marker/')
def segmentationMarker():
    segmentation = flask.request.values.get('segmentation')
    characters = []
    for testing in [True, False]:
        for item in data.listData(testing):
            segmentations = data.getSegmentations(item, segmentation)
            for character in segmentations:
                if character['status'] in ['INFERRED', 'OK', 'BAD']:
                    character['url'] = '/segment-image/%s/%s/%s' % (
                        item['id'],
                        segmentation,
                        character['label'],
                    )
                    characters.append(character)
    characters = sorted(
        characters, 
        key=lambda x: (0 if x['status'] == 'INFERRED' else 1), 
    )

    return flask.render_template('segmentationMarker.jinja2', 
        charactersJson=json.dumps(characters),
        segmentationJson=json.dumps(segmentation),
    )

@APP.route('/segmentation-marker/', methods=['POST'])
def segmentationMarkerSave():
    info = json.loads(flask.request.values.get('info'))
    data.updateSegmentation(info)
    print info
    return "OK"

@APP.route("/image/<id>")
def image(id):
    path = data.getImagePath(id)
    with open(path) as f:
        return f.read()

@APP.route("/segment-image/<id>/<segmentation>/<label>/")
def segmentationImage(id, segmentation, label):
    info = data.getInfo(id)
    path = "%s/%s/%s.png" % (info['path'], segmentation, label)
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    APP.run(debug=True)