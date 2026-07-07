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

- [x] constraints automata
- [x] generation loop by token and byte evaluation

- [-] tests
  - [x] accuracy > 95%
  - [-] performance < 5 mn for 10 provided inputs 
  - [-] robustness (files permissions, )

- [-] debugging and observability
  - [x] DEBUG mode showing top-k best tokens and masking
  - [-] improved vizualisation : overall progress and stats

```

# Algorithm explanation

1. __GET__ _logits_
2. __GET__ _current state_
3. __FOR EACH__ token in _logits_
4.   __IF__ token is invalid for schema and _current state_
5.   __THEN__ set its logit to _negative infinity_
6.   __UPDATE__ _best token_
7. __RETURN__ _best token_


# Design decisions

## _Which global approach should be used to validate the output (constrained decoding) ?_

__regex pattern__ 

- Implementation : A regex represent the desired output schema

Example:
```python
import regex
number_rgx = regex.compile(r"-?[0-9+](.[0-9+])?$")
# test with
generated = "-3.0"
number_rgx.match(generated)
```

- Pros : 
  - accuracy
  - performance : an eval through DFA occurs in linear complexity (O(n))
  - relatively easy to maintain
- Cons : 
  - less adapted to nested structures (JSON, XML, code)

 __context free grammar__

- Pros : 
  - better suited than regex (and even state machine) for complex syntax, like nested structures.
- Cons : 
  - algorithmic complexity, depending on the chosen algorithm

__state machine__ (chosen option)

- Pros : 
  - accuracy : 
  - performance when manually appending text
- Cons : 
  - flexibility : requires a big refactor when schemas change
  - requires rigor when triggering state change

## _How to store the score of the next token ?_

__in place logit modification__

- Pros: 
  - easy to write
- Cons: 
  - easier to debug

__boolean mask__

- Pros : performance (with numpy)

# Possible extensions

- context-free grammars : BNF
- Trie : optimal data structure to identify valid token ids
- multi-model compatibility
- recodingthe tokenizer
- performance optimization with caching and or batching

# Performance analysis

| Criteria | Treshold | All vocabulary | Vocabulary filter |
| --- | ---- | ----------------- |----------------- |
| JSON validity | 100% | | |
| execution speed | < 300 s | 495 | |
| accuracy | > 90% | 100 | |


# Challenges faced

## Having a single source of truth for string comparisons
 
__Challenge__ : Detecting transitions in state machine relies on string comparison. However, BPE (byte pair encoding) tokenizers encoding does not always match UTF-8 (cf. supra). This can be a source of complexity in the code (when relating the token ID - which the state machine should exclusively operate on - with those two kinds of encoding : UTF-8 strings, BPE bytes), as well as latency (when converting on the fly).

__Mitigation__ : Building dictionaries once and for all. Tradeoff : Some latency on initialization

## Minute evaluation of state transition

__Challenge__ : Due to BPE encoding, a token can cover two states. For instance, on `EXPECT_FUN_NAME` state, when evaluating token `my_fun", parameters:"` : `my_fun"` could cover the end of function name and `, parameters:` would already cover next state. We should be able to either refuse it for a smaller token covering only current state (with the risk of slowing down generation or even worth running out of acceptable tokens), or accepting it but advancing the state adequately.

__Mitigation__ : evaluating token at byte level. Tradeoff : Performance.

## Balancing accuracy and speed

__Challenge__ : Adapting the granularity level on which to apply constraints (byte, character, token) proved difficult. Checking every character before consuming the token ensures accuracy, at the cost of performance. Moreover, it does not fully leverage on the possibilities of define state automaton.

__Mitigation__ : 

- prefilter tokens with numpy before iterating over them. Tradeoff : Accuracy

## Grokking the theory

__Challenge__ : It took some time before distinguishing the big picture among all the new concepts introduced by the subject. 


# Testing strategy

## Improving debugging

`rich` library proved useful to build a table of logits (proposed and filtered) that could be printed on DEBUG mode.

## Integration tests for provided inputs

Tests are run using `unittest`. A map of expected output was added in regard to each inputs prompt.

## Unit tests for key functions

To be done (`utils/convert`)

## Manual robustness tests

Partially done for missing files, file permissions. Could be converted to unit test.

# Resources

| Url | Kind | Note              |
| --- | ---- | ----------------- |
|     |      |                   |
|[Andrew Docherty - Controlling your LLM](https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20)|🗞️ article|                   |
|[Aidan Cooper - A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)|🗞️ article |3 forms (regex, code, hybrid) + pitfalls |
|[Argparse doc](https://docs.python.org/3/library/argparse.html) | 📔 doc | module to parse arguments |
|[Pydantic doc](https://pydantic.dev/docs/validation/latest/concepts/models/)|📔 doc|Models used to validate input|
|[Qwen 0.6B on HF](https://huggingface.co/Qwen/Qwen3-0.6B)|      |                   |
|[Qwen 0.6B on APXML](https://apxml.com/models/qwen3-0-6b)|      |                   |
|[BPE Wikipedia](https://en.wikipedia.org/wiki/Byte-pair_encoding)|📙 wikipedia article|                   |

## Libraries usage

### argparse

- `ArgumentParser` holds the CLI configuration. Flags, types, default values are declared with `add_argument()`

### json module

- `load(filepath)` : transform a textfile with json into a python object (dict or list)
- `dumps(obj)` : convert tree into a string

### numpy

Optimize multidimensional array computation

- `np.argmax(array)` find index of maximum value. Used to select best candidate token
- `np.argsort(array)` : return indices that would sort array in asc. order. Used for debugging and visualization
- `np.full_like` generate a NumPy array with same shape and type as the one provided. Used to create a mask
- `np.isclose(a, b)` safely compare float arrays. Used for debugging and visualization

### pydantic

Required by the subject. Ensure schema validation.

- `BaseModel` helps declare required fields and their types
- `Field` to add metadata or behavior (ex : default value)
- `TypeAdapter` can be used to validate pydantic model against python types with `validate_python(json)`

### rich

Enrich display and logs

- `logging.RichHandler` can be integrated into `logging` module. Replace raw text with colored lines + file and line number
- `table.Table` to dynamically generate table
- `print` as an overload to native method 

## AI Usage

- __concept exploration__ and __documentation__(_define following concepts, disambiguate their relationships, provide an ontology or knowledge graph relative to following themes_)
- __guidance__ (_without giving code, define steps to reach this goal_) used to : 
  - learn how to setup a python project with new tools (uv)
  - learn rich syntax
- __code generation__ for repetitive tasks, as well as to provide a first example of a working setup when none was discovered (ie : state checking in a state machine)
- __feedback on code__ on generated code (_evaluate code quality and assess performance, security. If it does not meet requirements, prioritize tasks for its refactoring_)
- __debugging__ (_what is the cause of such error ? how to debug ? is it A, B or another ?_)

# Example usage

```
# make install

# make run

# make test

# make lint
```

# Concepts and Glossary

| Term | Def | Note              |
| --- | ---- | ----------------- |
|logit|raw score for a probable next token|usually spanning from -20 to +20|
|vocabulary|complete sets of unique tokens that a LM is trained to recognize and generate|Each token is mapped to a token ID|
|tensor|multidimensional container for numerical data. Can represent scalars (0 dimension), vectors (1D) and matrices (2D) to higher dimensions (N-D).|In a model, they store weights, input and other layers such as logits during inference|
|BPE (Byte pair encoding) tokenizer|Subword tokenization algorithm that merges most frequent pairs of adjacent characters or bytes into new tokens until a |Byte-level BPE initiates the process from 256 possible raw bytes values to ensure that any arbitrary binary flow can be tokenized without generating an unknown token error|
|ChatML|markup language developed by OpenAI and used in other models (Qwen) to structure a conversation|`<|im_start|>` indicates a new message and followed by sender role (`system`, `user`, `assistant`)|
|FSM - Finite state machine|moel consisting of a finite number of states, transitions and actions|In constrained decoding, it blocks invalid transitions|
|Trie or prefix tree|a search tree data structure storing an associative array where keys are equences|A trie allows to identify which token ID matches a valid string prefix allowed by the state machine|


## State machines

There are two kinds of state machines or automata:

- __finite state machines__ rely on strict linear transitions to validate _predictable_ patterns (i.e. known function signatures). They keep no memory of states before and beyond the current one.
- __push down automata__ work on LIFO basis. They can validate a __context-free grammar__.

During constraint decoding, state machines evaluate tokens or bytes against authorized transition from the current state. Sometimes, the transition is linear (ex: `EXPECT_PARAM_KEY` -> `EXPECT_PARAM_VALUE`). Sometimes, there are multiple possible states to take into account (ex: `EXPECT_COMMA_OR_END` -> `EXPECT_PARAM_KEY` OR `FINISH`)

## CFG - Context free grammar

CFG describe a syntaxis using relation between symbols. Some symbols are _terminal_ and represent the basic units (eg. digit or even bytes). Some are _non-terminal_ and are built upon them (eg float consisting possibly of a sequence of digits, dot, digits). cf. CFG in [JSON RFC 4627](https://www.rfc-editor.org/info/rfc4627/#section-2.2).

Most notable types of CFG families are
- LL : which we can read from left to right, with leftmost derivation (the leftmost non terminal symbol is replaced first)
- LR : which we can read from left to right, with inversed rightmost derivation. They are also called "bottom-up" as they start from terminal symbols.

A simplified example of grammar for the project would be:

```txt
output            = begin_object
                    json_prompt value_separator
                    json_name value_separator
                    json_parameters
                    end_object

json_prompt       = '"prompt" name_separator string
json_name         = '"name"' name_separator string
json_parameters   = '"parameters"' name_separator parameter_object
parameter_object  = begin_object [ member { value_separator member } ] end_object
member            = string name_separator value
value             = json_number | json_string | json_boolean
json_number       = [ "-" ] digit { digit } [ "." digit { digit } ] 
json_string       = '"' { char } '"' 
json_boolean      = "true" | "false" 

begin_object     = "{"
end_object       = "}"
name_separator   = ":"
value_separator  = ","
string           = '"' { char } '"'
char             = upperchar | lowerchar | digit | "_"
upperchar        = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
lowerchar        = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
digit            = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

```

Extended Backus-Naur form
```
=  : strictly
[] : optional
{} : 0 or more
A|B|C : altermatives
```

![railroad diagram](data/railroad_bnf.png)
_railroad diagram_ representing this CFG

__derivation__ refers to the replacement pattern of non-terminal symbols when constructing a valid sentence.
__determinist__ grammars can be parsed by an automaton without any ambiguity. 

DFA - Deterministic Finite Automaton : a symbol can lead to one and only one transition
NFA - Non deterministic Finite Automaton : a symbol can lead to many simultaneous transitions (ex: if the order of named parameter is not deemed important)
PDA - Pushdown Automaton : based on a heap. transition is guided not only by current symbol, but also by the one at the top of the heap. Can handle infinite nested structures.


## Byte-pair encoding tokenization

Encode _most frequent pairs of adjacent tokens_ into a new token, until the _vocabulary_ reaches a certain size.

While classical BPE starts from a character, _byte-level BPE_ (first used by GPT-2, then other models such as Qwen) builds upon raw bytes (ranging from 0 to 255). BPE allows to tokenize any binary flow (code, pictures, ...) without labelling any token as unknown.

__Tokenization example__

Starting from `"Hi 😄!"`

- Text to bytes
Each character becomes a byte (or series of bytes):
  - `H` -> `\x48`
  - `i` -> `\x69`
  - ` ` -> `\x20`
  - `😄`-> `\xf0\x9f\x98\x84`
  - `!` -> `\x21`

- Bytes to printable Unicode
To prevent control characters and space from breaking regexes or internal processes, we have to ensure that every possible byte value (0 to 255) is interpretable as a printable character.
  - `\x20` (32 in decimal value) -> `\x120` (Ġ, 288 in decimal value)

cf. function `bytes_to_unicode` in the project : it simply adds 256 to non printable characters.

- Grouping
Pairs are iteratively merged (most frequent are merged first), till the vocabulary reaches its max size (151 643 tokens for Qwen-2)
  - `\x48` (H) + `\x68` (i) -> token ID (ex 1234)

# Appendix

## Qwen 0.6B model identity

- 500 Mns parameters
- vocabulary size of 151,936

## Current usage of constrained decoding

_most of those tools were not authorized by the subject. Therefore it is just an overview for the sake of knowing current usage in the professional sphere_

- llama.cpp : supports CFG decoding
- SGLang : supports regex contraints, control primitives (`select`, `gen` 
- vLLM : cf [doc](https://vllm.ai/blog/2025-01-14-struct-decode-intro)

### other tools

- DSPy : optimizes prompts and examples to get closer to expected output
