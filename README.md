**py-jscrambler-deobfuscator**
Deobfuscator for JScrambler's Enterprise obfuscation written entirely in Python

#### Credits
Python port of the original https://github.com/Ciarands/jscrambler-deobfuscator

## Features
-  eval unpacking
-  control flow unflattening
-  code lock call removal
-  guard removal
-  state machine resolution
-  obfucated object restoration
-  vm lifting
-  string decoding
-  propagation, folding, dce
-  runtime cleanup

## Requirements

```shell
pip install esprima2 escodegen
```

## Usage

Run the deobfuscator against files in the samples folder with:

```shell
python3 deobfuscator.py
```

Commented out in deobfuscator.py is the 'LLM identifier renaming' transformer. This is a slow pass, but worthwhile if you'd like much better identifier names. The following is what I tested with for example:

```shell
OPENAI_API_KEY="none" ./llama-server \
  --hf-repo Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --hf-file qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  --n-gpu-layers 28 \
  -c 32768 \
  -n -1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --port 8080 \
  --host 127.0.0.1 \
  --alias qwen2.5-coder
```

#### Disclaimer
LLMs were used when developing this project. The jsir library and lifter transformer are both almost entirely generated.

This project has reached it's goal of fully removing the obfuscation runtime from all samples but is still in very early development. All transformers and the main loop need refactoring/redesign and testing.

This port is probably slower than the original. Any scope implementation is going to be slow, but this one does not track mutations. Scope has to be rebuilt constantly and that can be expensive. The other issue is that constant propagation, even partial, is very expensive.
