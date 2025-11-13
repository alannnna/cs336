"""
Hello!
"""
from collections import defaultdict
from typing import Generator, Any


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
    assert len(subt) == 2 # Code is written gneerically but only tested on len 2
    for i in range(len(t)-len(subt)+1):
        for subi in range(len(subt)):
            if t[i + subi] != subt[subi]:
                break
            if subi == len(subt)-1:
                return i
    return -1


def merge_bp(tokens: dict[tuple[bytes, ...], int], byte_pair: tuple[bytes, bytes]):
    merged = False
    merged_bp = (byte_pair[0] + byte_pair[1],)
    for token in list(tokens.keys()):
        found = tuple_contains(token, byte_pair)
        if (found >= 0):
            merged = True
            new_token = token[0:found] + merged_bp + token[found+2:]
            #print(f"replacing {token} with {new_token}")
            tokens[new_token] = tokens[token]
            tokens.pop(token)
    return tokens, merged


def main():
    corpus = """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest"""

    special_token = "<|endoftext|>"

    vocab = { i : bytes([i]) for i in range(256) }
    next_vocab_ix = 256
    vocab[next_vocab_ix] = special_token.encode("utf-8") # should this be encoded? does it matter?
    next_vocab_ix += 1

    pretokens = pretokenize(corpus)

    for _ in range(12):
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

        pretokens, merged = merge_bp(pretokens, max_pair)
        if merged:
            vocab[next_vocab_ix] = max_pair[0] + max_pair[1]
            next_vocab_ix += 1
        print(max_pair)
        print(pretokens)

    #print(vocab)


if __name__ == '__main__':
    main()
