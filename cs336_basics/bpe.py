"""
Hello!
"""
import os
from collections import defaultdict
from typing import Generator, Any


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError


def pretokenize(corpus: str) -> dict[tuple[bytes, ...], int]:
    counts = defaultdict(int)
    # split up string by whitespace, then encode each pretoken, then count up into a dictionary
    pretokens = corpus.split()
    for pt in pretokens:
        counts[tuple(bytes([b]) for b in pt.encode("utf-8"))] += 1
    return dict(counts)


def byte_pairs(data: tuple[bytes, ...]) -> Generator[tuple[bytes, bytes], Any, Any]:
    """Yield consecutive pairs of bytes objects from a tuple of bytes objects."""
    for i in range(len(data)-1):
        yield (data[i], data[i+1])


def tuple_contains(t: tuple, subt: tuple) -> int:
    for i in range(len(t)):
        for subi in range(len(subt)):
            if t[i] != subt[subi]:
                continue
            elif subi == len(subt)-1:
                return i
    return -1


def merge_bp(tokens: dict[tuple[bytes, ...], int], byte_pair: tuple[bytes, bytes]):
    for token in list(tokens.keys()):
        found = tuple_contains(token, byte_pair)
        if (found >= 0):
            merged_bp = (byte_pair[0] + byte_pair[1],)
            new_token = token[0:found] + merged_bp + token[found + len(byte_pair):-1]
            tokens[new_token] = tokens[token]
            tokens.pop(token)
    return tokens


def main():
    corpus = """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest"""

    special_token = "<|endoftext|>"

    vocab = { i : bytes([i]) for i in range(256) }
    vocab[256] = special_token.encode("utf-8")

    pretokens = pretokenize(corpus)

    for _ in range(6):
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        # iterate through pretokens
        for pt in pretokens:
            for bp in byte_pairs(pt):
                pair_counts[bp] += pretokens[pt]
        # Find max pair
        max_count = 0
        max_pair = (b'\x00', b'\x00')
        for pair, count in pair_counts.items():
            if count > max_count:
                max_count = count
                max_pair = pair
            if count == max_count and pair > max_pair:
                max_pair = pair

        pretokens = merge_bp(pretokens, max_pair)
        print(max_pair)
        print(pretokens)


if __name__ == '__main__':
    main()
