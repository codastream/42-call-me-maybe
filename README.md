_This project has been created as part of the 42 curriculum by fpetit_

# Description

This project aims at implementing following SLM optimizations : 

- __function calling__ : translating natural language request into a function call
- __constrained decoding__ to ensure generated output follows required format (here JSON)

# Progress

```md
- [x] validate input and output
- [x] Pydantic schemas
- [x] extract vocabulary

- [ ] constraints automata
  - [ ] NAME : authorize only tokens completing function names
  - [ ] TRANSITION : force transition string `", "parameters": {`
  - [ ] KEYS : authorize only tokens completing param keys associated with chose function
  - [ ] VALUES : authorize only tokens according to type definition

- [ ] generation loop token by token
  - call `get_logits_from_input_ids`
  - get acceptable tokens list from automata for current state
  - apply mask (- inf)
  - select optimum token and add it to history
  - stop loop wgen final char `}` or EOS token

- [ ] tests
  - evaluate precision

```

# Instructions

```bash
uv sync
make run
make debug
make lint
```

# Resources

| Url | Kind | Note              |
| --- | ---- | ----------------- |
|     |      |                   |
|[A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)|🗞️ article |3 forms (regex, code, hybrid) + pitfalls |
| https://docs.python.org/3/library/argparse.html | 📔 doc | module to parse arguments |
|[Pydantic doc](https://pydantic.dev/docs/validation/latest/concepts/models/)|📔 doc|Models used to validate input|
|[Qwen 0.6B on HF](https://huggingface.co/Qwen/Qwen3-0.6B)|      |                   |
|[Qwen 0.6B on APXML](https://apxml.com/models/qwen3-0-6b)|      |                   |
|[BPE Wikipedia](https://en.wikipedia.org/wiki/Byte-pair_encoding)|📙 wikipedia article|                   |

## Useful functions

- `np.argmax` : find index of maximum value - used to select best candidate token
- `np.full_like` : generate a NumPy array with same shape and type as the one provided

## AI Usage

- Pedagogical prompt (give no code, introduce and define new concepts progressively) with Gemini to help learn how to setup a python project with new tools (uv)

# Qwen 0.6B model identity

- vocabulary size of 151,936

## How constrained decoding is actually used + other current tools

llama.cpp : supports CFG decoding
SGLang : supports regex contraints, control primitives (`select`, `gen` 
vLLM : cf [doc](https://vllm.ai/blog/2025-01-14-struct-decode-intro)

### other tools
DSPy : optimizes prompts and examples to get closer to expected output

# Glossary

| Term | Def | Note              |
| --- | ---- | ----------------- |
|logit|raw score for a probable next token|usually spanning from -20 to +20|
|vocabulary size|      |      |
|tensor|      |      |
|BPE tokenizer|Byte pair encoding|      |
|gQA - Grouped query attention||      |
|SwiGLU activation||      |
|RoPE - Rotary Positional Embeddings||      |
|ChatML|markup language used by some models (OpenAI, Alibaba) to structure a conversation and distinguish roles|`<|im_start|>` indicates a new message and followed by sender role (`system`, `user`, `assistant`)|
|||      |
|||      |

## Byte-pair encoding

Encode _most frequent pairs of adjacent tokens_ into a new token, until the _vocabulary_ reaches a certain size.

Qwen uses _byte-level BPE_

Example tokenization from `"Hello world 😄 !"`

- Text to bytes
Each character becomes a byte (or series of bytes):
😄 -> `\xf0\x9f\x98\x84`

- Visual mask
When decoding bytes into UTF-8, it could generate spaces and non printable characters. This could break terminal display or file integrity. Therefore, we have to ensure that every possible byte value (0 to 255) is interpretable as a printable character (cf. function `bytes_to_unicode` in the project).
`b' '` -> `Ġ`

- Grouping
Pairs are merged (most frequent are merged first), till the vocabulary reaches its max size (151 643 tokens for Qwen-2)
`Hello` -> ID


# Algorithm explanation

```md
__GET__ _logits_
__GET__ _current state_
__FOR EACH__ token in _logits_
  __IF__ token is invalid for schema and _current state_
  __THEN__ set its logit to _negative infinity_
  __UPDATE__ _best token_
__RETURN__ _best token_
```

# Design decisions

_How to validate the next token (constrained decoding) ?_

- regex pattern representing the schema and checking for a partial match
  - __pros__ : relatively easy to read and modify
  - __cons__ : performance (especially if the model has a big vocabulary)
- state machine (ex : EXPECTING_KEY -> EXPECTING_FUN_NAME -> EXPECTING ARGS)
  - __pros__ : performance, especially when managing transition (as we then manually complete the structure, complexity is O(1))
  - __cons__ : requires rigor to detect when to change state
- context free grammar (in Backus-Naur form)
  - __pros__ : well-suited for complex syntax
  - __cons__ : algorithmic complexity

_How to store the score of the next token ?_
- in place loop
  - __pros__ : easier to debug
- boolean mask
  - __pros__ : performance (with numpy)

# Possible extensions

Context-free grammars : BNF : alternative to regex
Trie : optimal data structure to identify valid token ids

# Performance analysis

- JSON validity == 100% ?
- execution speed < 5mn ?
- accuracy > 90% ?

# Challenges faced

- cleaning tokenizer special chars : BPE tokenizers include special characters. For instance : `Ġ` to indicate a space. Substring encoding depends of its position. They should be cleaned before checking matches. 
- detecting state change
- dynamic constraints
- decoding performance
- decoding reliability

| Challenge | Mitigation |
| --- | ---- |
|     |      |

# Testing strategy

- nominal cases
- syntaxic resilience (special characters, ambiguous prompts)
- check against expecting result

# Example usage

