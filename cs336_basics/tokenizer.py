from typing import Iterator, Iterable, Self
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> Self:
        vocab = {}
        with open(vocab_filepath, "r") as f:
            for line in f:
                k, _, v = line.strip().partition(",")
                if not v:
                    print(f"WARN: line <<{line.strip()}>> couldn't be partitioned into k,v")
                vocab[eval(k)] = eval(v)

        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                merges.append(eval(line.strip()))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pretokenize
        pretokens = []
        # TODO handle special_tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, text):
            pretokens.append(tuple(bytes([b]) for b in match.group().encode("utf-8")))
        # merge
        # TODO
        return []

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for n in self.encode(s):
                yield n

    def decode(self, ids: list[int]) -> str:
        bits = []
        for id in ids:
            bits.append(self.vocab[id])
        return b"".join(bits).decode("utf-8", errors="replace")


def main():
    t = Tokenizer.from_files("sample-vocab.txt", "sample-merges.txt", ["<|endoftext|>"])
    from IPython import embed; embed()


if __name__ == "__main__":
    main()
