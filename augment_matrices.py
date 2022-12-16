import random
import math


def get_stretch_matrix(stretchCoefficient=0.2):
    stretchFactor = random.uniform(-stretchCoefficient, stretchCoefficient)

    horizontalStretchFactor = 1 + stretchFactor
    verticalStretchFactor = 1 - stretchFactor

    v1 = [horizontalStretchFactor, 0, 0]
    v2 = [0, verticalStretchFactor, 0]

    return [v1, v2, [0, 0, 1]]


def get_shear_matrix(shearCoefficient=0.5):

    v1 = [1, (random.random() - 0.5) * shearCoefficient, 0]
    v2 = [(random.random() - 0.5) * shearCoefficient, 1, 0]

    return [v1, v2, [0, 0, 1]]


def get_zoom_matrix(zoomCoefficient=0.2):
    zoomFactor = random.uniform(1 - zoomCoefficient, 1 + zoomCoefficient)

    v1 = [zoomFactor, 0, 0]
    v2 = [0, zoomFactor, 0]

    return [v1, v2, [0, 0, 1]]


def get_flip_matrix():

    v1 = [1 + -2 * random.randint(0, 1), 0, 0]
    v2 = [0, 1 + -2 * random.randint(0, 1), 0]

    return [v1, v2, [0, 0, 1]]


def get_rotate_matrix(only_rotate_90=True):
    if only_rotate_90:
        angle = random.randint(0, 3) * math.pi / 2
    else:
        angle = random.randint(0, 359)

    v1 = [math.cos(angle), math.sin(angle), 0]
    v2 = [-math.sin(angle), math.cos(angle), 0]

    return [v1, v2, [0, 0, 1]]


def get_translate_matrix(maxTranslateRange):
    v1 = [1, 0, (random.random() - 0.5) * maxTranslateRange]
    v2 = [0, 1, (random.random() - 0.5) * maxTranslateRange]

    v3 = [0, 0, 1]

    return [v1, v2, v3]
