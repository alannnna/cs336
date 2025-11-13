import os
from collections import defaultdict
import regex as re
from pretokenization_example import find_chunk_boundaries


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
    max_file_size = 5_000_000
    with open(input_path, "rb") as f:
        if os.path.getsize(input_path) > max_file_size:
            print(f"WARN: file was {os.path.getsize(input_path)}, only reading first 5MB")
            chunk = f.read(max_file_size).decode("utf-8", errors="ignore")
        else:
            chunk = f.read().decode("utf-8", errors="ignore")
        """
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            break
        """

    counts = defaultdict(int)
    special_tokens_pattern = '|'.join(re.escape(token) for token in special_tokens)
    docs = re.split(special_tokens_pattern, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in docs:
        for match in re.finditer(PAT, doc):
            counts[tuple(bytes([b]) for b in match.group().encode("utf-8"))] += 1

    for k, v in list(counts.items())[:10]:
        print(k, v)

    return {}, []


def main():
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 512, special_tokens)
    print(vocab)
    print(merges)


if __name__ == "__main__":
    main()
