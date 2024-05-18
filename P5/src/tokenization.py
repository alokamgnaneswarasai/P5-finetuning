from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
import re
import sentencepiece as spm

print("P5/src/tokenization.py")

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# I create another class P5Tokenizer extending it to add <user_id_{}> & <item_id_{}>

class P5Tokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        user_extra_ids=0,
        item_extra_ids=0,
        additional_special_tokens=None,
        legacy = False,
        add_prefix_space = False,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if user_extra_ids > 0:
            additional_special_tokens.extend(["<user_id_{}>".format(i) for i in range(user_extra_ids)])
        
        if item_extra_ids > 0:
            additional_special_tokens.extend(["<item_id_{}>".format(i) for i in range(item_extra_ids)])


        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._user_extra_ids = user_extra_ids
        self._item_extra_ids = item_extra_ids
        self.sp_model = spm.SentencePieceProcessor()   
        self.sp_model.Load(vocab_file)
        self.legacy = legacy
        self.add_prefix_space = add_prefix_space

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        
    @property
    def vocab_size(self):
        
       
        # return spm.SentencePieceProcessor().get_piece_size() + self._extra_ids + self._user_extra_ids + self._item_extra_ids
        return self.sp_model.get_piece_size() + self._extra_ids + self._user_extra_ids + self._item_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._user_extra_ids - self._item_extra_ids
        elif "<user_id_" in token:
            match = re.match(r"<user_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._item_extra_ids
        elif "<item_id_" in token:
            match = re.match(r"<item_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids + self._user_extra_ids - 1:
                token = "<item_id_{}>".format(self.vocab_size - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<user_id_{}>".format(self.vocab_size - self._item_extra_ids - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self._user_extra_ids - self._item_extra_ids - 1 - index)
        return token


# Below are for Rust-based Fast Tokenizer
from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from typing import Any, Dict, List, Optional, Tuple, Union


class P5Converter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]

        num_user_extra_ids = self.original_tokenizer._user_extra_ids
        vocab += [("<user_id_{}>".format(i), 0.0)
                  for i in range(num_user_extra_ids - 1, -1, -1)]
        
        num_item_extra_ids = self.original_tokenizer._item_extra_ids
        vocab += [("<item_id_{}>".format(i), 0.0)
                  for i in range(num_item_extra_ids - 1, -1, -1)]

        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_p5tokenizer(p5tokenizer):
    return P5Converter(p5tokenizer).converted()


class P5TokenizerFast(T5TokenizerFast):
    slow_tokenizer_class = P5Tokenizer

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        user_extra_ids=50,
        item_extra_ids=50,
        additional_special_tokens=None,
        add_prefix_space=False,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )
            
        if user_extra_ids > 0:
            additional_special_tokens.extend(["<user_id_{}>".format(i) for i in range(user_extra_ids)])
        
        if item_extra_ids > 0:
            additional_special_tokens.extend(["<item_id_{}>".format(i) for i in range(item_extra_ids)])

        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            user_extra_ids=user_extra_ids,
            item_extra_ids=item_extra_ids,
            **kwargs
        )
        fast_tokenizer = convert_slow_p5tokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer

        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            user_extra_ids=user_extra_ids,
            item_extra_ids=item_extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._user_extra_ids = user_extra_ids
        self._item_extra_ids = item_extra_ids
        
        self.add_prefix_space = add_prefix_space
        self.sp_model = spm.SentencePieceProcessor()
