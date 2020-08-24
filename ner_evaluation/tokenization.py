from typing import List, Tuple

from pytorch_transformers.tokenization_bert import (
    _is_punctuation as is_punctuation,
    _is_whitespace as is_whitespace,
)


class Token(object):
    """Info about a single token."""

    def __init__(self,
                 text: str,
                 offset: int,
                 index: int,
                 tail: str = '',
                 tag: str = None):

        if not isinstance(text, str) or not text:
            raise TypeError('text should be a non-empty string.')
        if not isinstance(offset, int) or offset < 0:
            raise TypeError('offset should be an int >= 0.')
        if not isinstance(index, int) or index < 0:
            raise TypeError('index should be an int >= 0.')

        self.text = text
        self.offset = offset
        self.tail = tail
        self.tag = tag
        self._example = None
        self._index = index

    def __str__(self):
        return '{}{}'.format(self.text, self.tail)

    def __repr__(self):
        return 'Token(text=%r, offset=%r, index=%r, tail=%r, tag=%r)' % \
            (self.text, self.offset, self.index, self.tail, self.tag)

    def __len__(self):
        return len(self.text) + len(self.tail)

    def __add__(self, char):
        self.text += char
        return self

    @property
    def example(self):
        return self._example

    @property
    def index(self):
        return self._index

    @property
    def is_punct(self):
        return is_punctuation(self.text)

    def has_tail(self):
        return bool(self.tail)

    @property
    def nbor(self):
        """Returns the neighboring token, e.g., 
        self._example.doc_tokens[self.index + 1]."""
        if self.index is None:
            return None
        try:
            return self._example.doc_tokens[self.index + 1]
        except IndexError:
            return None


def reconstruct_text_from_tokens(tokens: List[Token],
                                 include_last_tail: bool = False,
                                 ) -> str:
    """Concatenates the text of a sequence of tokens."""
    def text_generator(tokens):
        for i, token in enumerate(tokens):
            yield token.text
            if i < len(tokens) - 1 or include_last_tail:
                yield token.tail

    return ''.join(piece for piece in text_generator(tokens))


class TokenizerWithAlignment:
    """Tokenizer that performs basic tokenization keeping string alignment."""

    def __init__(self):
        pass

    @staticmethod
    def _begin_new_token(doc_tokens, text, offset):
        token = Token(text=text, offset=offset, index=len(doc_tokens))
        doc_tokens.append(token)

        return token

    def tokenize(self, text: str) -> Tuple[List[Token], List[int]]:
        doc_tokens = []
        char_to_word_offset = []

        new_word = True
        curr_token = None

        for offset, c in enumerate(text):
            if is_whitespace(c):
                new_word = True
                if curr_token:
                    curr_token.tail += c
            else:
                if is_punctuation(c):
                    curr_token = self._begin_new_token(doc_tokens, c, offset)
                    new_word = True
                else:
                    if new_word:
                        curr_token = self._begin_new_token(
                            doc_tokens, c, offset)
                    else:
                        curr_token += c
                    new_word = False

            # OBS: Whitespaces that appear before any tokens will have offset -1
            # char_to_word_offset.append(len(doc_tokens) - 1)
            char_to_word_offset.append(max(0, len(doc_tokens) - 1))

        return doc_tokens, char_to_word_offset

    def __call__(self, text: str) -> Tuple[List[Token], List[int]]:
        return self.tokenize(text)
