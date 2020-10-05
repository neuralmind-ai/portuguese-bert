"""
Defines NER tag encoder for predefined coding schemes.
"""
from typing import List

BIO = 'BIO'
BILUO = 'BILUO'

SCHEMES = {
    BIO: ['B', 'I'],
    BILUO: ['B', 'I', 'L', 'U'],
}

VALID_TRANSITIONS = {
    BIO: {
        'B': ['B', 'I', 'O'],
        'I': ['B', 'I', 'O'],
        'O': ['B', 'O'],
    },
    BILUO: {
        'B': ['I', 'L'],
        'I': ['I', 'L'],
        'L': ['B', 'U', 'O'],
        'U': ['B', 'U', 'O'],
        'O': ['B', 'U', 'O'],
    },
}


class NERTagsEncoder(object):
    """Handles creation of NER tags for a list of named entity classes and
    conversion of tags to ids and vice versa."""

    def __init__(self,
                 classes: List[str],
                 scheme: str = BIO,
                 ignore_index: int = -100):

        if not len(set(classes)) == len(classes):
            raise ValueError("`classes` have duplicate entries.")
        if "O" in classes or "X" in classes:
            raise ValueError("`classes` should not have tag O nor X.")
        if ignore_index >= 0 or not isinstance(ignore_index, int):
            raise ValueError("`ignore_index` should be a negative int.")
        if scheme not in SCHEMES:
            raise ValueError("`scheme` should be one of {}".format(
                tuple(SCHEMES.keys())))

        self.classes = tuple(classes)
        self.tags = ["O"]
        self.ignore_index = ignore_index
        self.tag_to_id = {"X": ignore_index}
        self.scheme = scheme

        for clss in classes:
            for subtag in SCHEMES[scheme]:
                self.tags.append(f"{subtag}-{clss}")

        for i, tag in enumerate(self.tags):
            self.tag_to_id[tag] = i

    def __repr__(self):
        return ('{class_}(classes={classes!r}, scheme={scheme!r})') \
            .format(class_=self.__class__.__name__,
                    classes=self.classes,
                    scheme=self.scheme)

    @classmethod
    def from_labels_file(cls, filepath: str, *args, **kwargs):
        """Creates encoder from a file with NER label classes (one class per
        line) and a given scheme."""
        with open(filepath, 'r') as fd:
            ner_classes = [clss for clss in fd.read().splitlines() if clss]

        return cls(ner_classes, *args, **kwargs)

    @property
    def num_labels(self) -> int:
        return len(self.tags)

    def convert_tags_to_ids(self, tags: List[str]) -> List[int]:
        """Converts a list of tag strings to a list of tag ids."""
        return [self.tag_to_id[tag] for tag in tags]

    def convert_ids_to_tags(self, tag_ids: List[int]) -> List[str]:
        """Returns a list of tag strings from a list of tag ids."""
        return [self.tags[tag_id] for tag_id in tag_ids]

    def decode_valid(self, tag_sequence: List[str]) -> List[str]:
        """Processes a list of tag strings to remove invalid predictions given
        the valid transitions of the tag scheme, such as "I" tags coming after
        "O" tags."""
        if self.scheme == BILUO:
            import warnings
            warnings.warn(f"Valid decoding for BILUO scheme is not implemented. Returning input sequence.")
            return tag_sequence

        prev_tag = 'O'
        prev_type = 'O'

        final = []
        for tag_and_cls in tag_sequence:
            tag = tag_and_cls[0]
            type_ = tag_and_cls.split('-')[-1]
            valid_transitions = VALID_TRANSITIONS[self.scheme][prev_tag]

            valid_tag = False
            if tag in valid_transitions:
                if tag in ('B', 'O'):
                    valid_tag = True
                elif tag == 'I' and type_ == prev_type:
                    valid_tag = True

            if valid_tag:
                prev_tag = tag
                prev_type = type_
                final.append(tag_and_cls)
            else:
                prev_tag = 'O'
                prev_type = 'O'
                final.append('O')

        return final