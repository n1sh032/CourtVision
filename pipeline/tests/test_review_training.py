import pytest
from courtvision.contact import detect, detect_for_review
from courtvision.contact_model import ContactModel, Label, Review, train
from courtvision.schema import ClipResult, ContactEvent, Frame, PlayerFrame, VideoMeta


def player(track, left, *, visible=True):
    points = [(0, 0, 0)] * 17
    points[9] = (45, 45, 1)
    if visible:
        points[5], points[6] = (left + 5, 25, 1), (left + 20, 25, 1)
        points[11], points[12] = (left + 5, 65, 1), (left + 20, 65, 1)
