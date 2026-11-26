import pytest
from courtvision.contact import detect, detect_for_review
from courtvision.contact_model import ContactModel, Label, Review, train
from courtvision.schema import ClipResult, ContactEvent, Frame, PlayerFrame, VideoMeta


def player(track, left, *, visible=True):
    points = [(0, 0, 0)] * 17
    points[9] = (45, 45, 1)
