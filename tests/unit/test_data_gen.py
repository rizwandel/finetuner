import pytest

from ..data_generator import (
    fashion_doc_generator,
    fashion_match_doc_generator,
    fashion_match_documentarray,
)

# import matplotlib.pyplot as plt


def test_doc_generator():
    for d in fashion_doc_generator():
        assert d.tags['class']
        break


@pytest.mark.parametrize('channels', [0, 1, 3])
@pytest.mark.parametrize('upsampling', [1, 2, 4])
def test_doc_generator_channel(channels, upsampling):
    for d in fashion_doc_generator(channels=channels, upsampling=upsampling):
        if channels == 0:
            assert d.blob.ndim == 2
        else:
            assert d.blob.ndim == 3
            assert d.blob.shape[-1] == channels

        assert d.blob.shape[0] == 28 * upsampling
        assert d.blob.shape[1] == 28 * upsampling
        # plt.imshow(d.blob, aspect='equal')
        # plt.show()
        break


@pytest.mark.parametrize('pos_value, neg_value', [(1, 0), (1, -1)])
@pytest.mark.parametrize('num_pos, num_neg', [(5, 7), (10, 10)])
def test_fashion_matches_generator(num_pos, num_neg, pos_value, neg_value):
    for d in fashion_match_doc_generator(
        num_pos=num_pos, num_neg=num_neg, pos_value=pos_value, neg_value=neg_value
    ):
        assert len(d.matches) == num_pos + num_neg
        all_labels = [int(d.tags['trainer']['label']) for d in d.matches]
        assert all_labels.count(pos_value) == num_pos
        assert all_labels.count(neg_value) == num_neg
        for m in d.matches:
            if int(m.tags['trainer']['label']) == 1:
                assert m.tags['class'] == d.tags['class']
            else:
                assert m.tags['class'] != d.tags['class']
        break


def test_fashion_documentarray():
    da = fashion_match_documentarray(num_total=10, num_pos=2, num_neg=3)
    assert len(da) == 10
    assert len(da[0].matches) == 5