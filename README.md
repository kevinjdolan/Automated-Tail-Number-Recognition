Automated-Tail-Number-Recognition
=================================

Automatically extract tail numbers from photos of U.S. aircraft.

## History

This was my attempt to start a company that used cameras and algorithms to provide analytical tools to uncontrolled local airports about flight activity. I came up with the idea while training to fly helicopters at such an airport in Linden, New Jersey.

I called a bunch of airports and nobody was interested. The algorithm didn't really work either.

And god, how simple it would be now with VLMs.

## Implementation

Pre-deep-learning classical computer vision, written in Python 2 against `scikit-image`, `numpy`, `scipy`, and `flask` — i.e., the kind of pipeline you write when you don't yet know that the next decade is going to make all of this a one-line model call. The plan:

1. **Threshold and label.** `AdaptiveThreshold` (`atnr/operations.py:138`) binarizes the image with a gaussian field size hand-tuned to the expected tail-area height, and `LabelConnectedComponents` (`atnr/operations.py:476`) assigns a label per blob. If the tail is glare-y, dirty, or in shadow — well, hopefully it isn't.
2. **Filter candidate segments.** `isReasonableSegment` (`atnr/util.py:18`) throws out anything too small, too large, or with an aspect ratio outside `LETTER_ASPECT_RATIO_MEAN ± 3σ`. Survivors get scored by `KnnPixelVectorCandidateExtractor` (`atnr/operations.py:464`), which squashes the segment into a fixed square, flattens it to a bool vector, and asks a KNN classifier whether that looks like a character. The classifier module isn't checked in. Of course.
3. **Group into a neighborhood.** `FindSegmentNeighborhood` (`atnr/operations.py:414`) walks the candidates and clusters any two whose relative position and size fall inside another `mean ± 3σ` envelope, this time over letter-pair statistics from `atnr/constants.py`. Whichever cluster ends up biggest is declared the tail. Ties go to whoever's first.
4. **Fill in the missing letters with a ruler.** When the cluster has fewer than `TAIL_NUMBER_LENGTH` (6) members — i.e., often — `FindNeighborhoodMask` (`atnr/operations.py:286`) does a linear regression on the visible letters' bbox coordinates and extrapolates where the missing ones *would* be if everything were planar, evenly spaced, and cooperative.
5. **Rectify.** `CorrectOrientation` (`atnr/operations.py:194`) takes the four corners of the (possibly hallucinated) neighborhood and applies a `ProjectiveTransform` to warp it into a flat rectangle, so that a hypothetical character recognizer can have a nice day.

Everything is built on a homemade `BaseOperation` framework with declared `__inputs__` / `__outputs__` and a shared environment dict, glued together by `ComposeOperations`. It is a small, earnest dataflow graph that exists because in 2014 it felt easier to write one than to find one.

The numerical constants in `atnr/constants.py` (letter aspect ratio, fill ratio, neighborhood overlap variance, vertical histogram min, and friends) were fit from the training set by some script that is no longer with us. The file opens with `# TODO: REPRODUCIBILITY!!`, which tells you everything.

## Data

`data/training/` and `data/testing/` each contain a directory per sample, named by UUID. Each contains:

- `image.png` — the source photo
- `info.json` — `{tail, characters, version, ...}` where `characters` is the list of click-tagged `[x, y]` pixel locations of each letter
- `candidate-segmentation/` — per-label PNGs and JSON produced by `atnr/candidateSegmentation.py`, which auto-tags any connected component that contains a known character location

There are ~273 training and ~103 testing samples.

## Tagging UI

`atnr/tagging.py` is a small Flask app for manually labeling the dataset:

- `/tail-character-locations/` — type the tail number and click each character's location in the image
- `/segmentation-marker/` — review auto-tagged candidate segments and mark them OK/BAD

Run with `python atnr/tagging.py` and visit `http://localhost:5000/`. Templates are in `atnr/templates/`.

## Status

Abandoned. Python 2 only, depends on an old `skimage.filter` (renamed to `skimage.filters` in later versions), and references a `knnPixel` module that isn't in the repo. Kept around as a historical artifact — see above.
