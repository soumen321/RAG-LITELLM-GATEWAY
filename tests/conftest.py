import os
import pytest
os.environ.setdefault("OPENAI_API_KEY",    "sk-test")
os.environ.setdefault("GATEWAY_API_KEY",   "dev-secret-key")
os.environ.setdefault("CHROMA_PERSIST_DIR","./data/test_chroma")