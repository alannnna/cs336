import os
import sys
from collections import defaultdict
import regex as re
from typing import Generator, Any


def pretokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    counts = defaultdict(int)
    cached_tuples = {}
    special_tokens_pattern = '|'.join(re.escape(token) for token in special_tokens)
    docs = re.split(special_tokens_pattern, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in docs:
        for match in re.finditer(PAT, doc):
            cached_tuple = cached_tuples.get(match.group(), None)
            if cached_tuple is None:
                cached_tuple = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                cached_tuples[match.group()] = cached_tuple
            counts[cached_tuple] += 1
    return counts


def byte_pairs(data: tuple[bytes, ...]) -> Generator[tuple[bytes, bytes], Any, Any]:
    """Yield consecutive pairs of bytes objects from a tuple of bytes objects."""
    for i in range(len(data)-1):
        yield (data[i], data[i+1])


def tuple_contains(t: tuple, subt: tuple) -> int:
    # assert len(subt) == 2
    for i in range(len(t)-1):
        for subi in range(2):
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
        while (found >= 0):
            merged = True
            new_token = token[0:found] + merged_bp + token[found+2:]
            #print(f"replacing {token} with {new_token}")
            tokens[new_token] = tokens[token]
            tokens.pop(token)
            token = new_token
            found = tuple_contains(token, byte_pair)
    return tokens, merged


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
    truncate_file = True
    max_file_size = 500_000_000
    with open(input_path, "rb") as f:
        if truncate_file and os.path.getsize(input_path) > max_file_size:
            print(f"WARN: file was {os.path.getsize(input_path)/1000/1000}MB, only reading first {max_file_size/1000/1000}MB")
            chunk = f.read(max_file_size).decode("utf-8", errors="ignore")
        else:
            chunk = f.read().decode("utf-8", errors="ignore")

    vocab = { i : bytes([i]) for i in range(256) }
    next_vocab_ix = 256
    for st in special_tokens:
        vocab[next_vocab_ix] = st.encode("utf-8") # should this be encoded? does it matter?
        next_vocab_ix += 1

    pretokens = pretokenize(chunk, special_tokens)
    #for k, v in list(pretokens.items())[:10]:
    #    print(k, v)

    merges = []
    print_counter = 0
    while len(vocab) < vocab_size:
        if (print_counter % 100 == 0):
            print(f"starting iter {print_counter}")
        print_counter += 1

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
        if not merged:
            print(f"WARN: no merge for pair {max_pair}. logic error?")
            break
        vocab[next_vocab_ix] = max_pair[0] + max_pair[1]
        next_vocab_ix += 1
        merges.append(max_pair)

        #print(max_pair)
        #print(pretokens)

    return vocab, merges


def main():
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", 1_000, special_tokens)
    print(vocab)
    print(merges)


if __name__ == "__main__":
    if (len(sys.argv) >= 2 and sys.argv[1] in ['-p', '--profile']):
        import cProfile
        cProfile.run('main()')
    else:
        main()
