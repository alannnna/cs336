import os
import sys
from datetime import datetime
from collections import defaultdict
import regex as re
from typing import Generator, Any
import multiprocessing as mp
from functools import reduce

from cs336_basics.pretokenization_example import find_chunk_boundaries


# TODO:
# [] seralize vocab and merges to disk
# [] train on OWT

def pretokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    counts = defaultdict(int)
    cached_tuples = {}
    special_tokens_pattern = '|'.join(re.escape(token) for token in special_tokens)
    docs = re.split(special_tokens_pattern, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in docs:
        for match in re.finditer(PAT, doc):
            mg = match.group()
            cached_tuple = cached_tuples.get(mg, None)
            if cached_tuple is None:
                cached_tuple = tuple(bytes([b]) for b in mg.encode("utf-8"))
                cached_tuples[mg] = cached_tuple
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


def read_chunk_and_pretokenize(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pretokens = pretokenize(chunk, special_tokens)
    return pretokens


# Note: mutates arguments!
def merge_pretoken_dicts(dicts: list[dict]) -> dict:
    if len(dicts) == 0:
        return {}
    ret = dicts[0]
    for d in dicts[1:]:
        for k, v in d.items():
            if k in ret:
                ret[k] += v
            else:
                ret[k] = v
    return ret


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
    # set up vocab
    vocab = { i : bytes([i]) for i in range(256) }
    next_vocab_ix = 256
    for st in special_tokens:
        vocab[next_vocab_ix] = st.encode("utf-8") # should this be encoded? does it matter?
        next_vocab_ix += 1

    serial = False
    truncate_file = True
    max_file_size = 500_000_000
    if serial:
        with open(input_path, "rb") as f:
            if truncate_file and os.path.getsize(input_path) > max_file_size:
                print(f"WARN: file was {os.path.getsize(input_path)/1000/1000}MB, only reading first {max_file_size/1000/1000}MB")
                chunk = f.read(max_file_size).decode("utf-8", errors="ignore")
            else:
                chunk = f.read().decode("utf-8", errors="ignore")
        pretokens = pretokenize(chunk, special_tokens)
    else:
        boundaries = []
        num_processes = os.cpu_count() or 4
        assert len(special_tokens) >= 1
        with open(input_path, "rb") as f:
            # Assume the first special token is okay to split chunks on...
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        if not boundaries:
            print("ERROR: couldn't find chunk boundaries")

        if truncate_file:
            new_boundaries = []
            for b in boundaries:
                if b < max_file_size:
                    new_boundaries.append(b)
                else:
                    new_boundaries.append(max_file_size)
                    print(f"truncated boundaries from {boundaries} to {new_boundaries}")
                    boundaries = new_boundaries
                    break

        with mp.Pool(num_processes) as p:
            args = [
                (input_path, start, end, special_tokens)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            pretoken_dicts = p.map(read_chunk_and_pretokenize, args)
        pretokens = merge_pretoken_dicts(pretoken_dicts)

    merges = []
    print_counter = 0
    while len(vocab) < vocab_size:
        if (print_counter % 100 == 0):
            print(f"starting iter {print_counter} t={datetime.now()}")
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
    print(f"Hello! t={datetime.now()}")
    special_tokens = ["<|endoftext|>"]
    max_vocab_size = 500
    vocab, merges = run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", max_vocab_size, special_tokens)

    # probably super inefficient, whatever
    with open("vocab.txt", "w") as f:
        for k, v in vocab.items():
            f.write(f"{k},{v}\n")
    with open("merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge}\n")


if __name__ == "__main__":
    if (len(sys.argv) >= 2 and sys.argv[1] in ["-p", "--profile"]):
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        main()

        profiler.disable()
        profiler.dump_stats("output.prof")
    else:
        main()
