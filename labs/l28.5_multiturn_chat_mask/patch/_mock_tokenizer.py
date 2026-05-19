"""MockTokenizer：模拟 Qwen / QwQ 类 chat template 的两种麻烦行为，便于 CPU 单元测试。

不要修改本文件；学生应只改 starter/multiturn_tokenizer.py。

行为开关：
- default_system_mode=True：当 messages 不以 system 开头时，模板会注入默认 system。
- qwq_mode=True：当 assistant 不是最后一条 message 时，模板会从其内容里删去 <think>...</think>。

encode 用 ASCII char-level 方案，方便测试断言。
"""

from __future__ import annotations

import re
from typing import List, Mapping


class MockTokenizer:
    """ASCII char-level mock tokenizer with controllable chat-template quirks."""

    def __init__(self, *, default_system_mode: bool = False, qwq_mode: bool = False) -> None:
        self.default_system_mode = default_system_mode
        self.qwq_mode = qwq_mode

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Each Unicode char → its codepoint as a token id (deterministic, decodable).
        return [ord(c) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) for i in ids if i >= 0)

    def apply_chat_template(
        self,
        messages: List[Mapping[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | List[int]:
        msgs = [dict(m) for m in messages]

        # Quirk 1: default system injection when none present.
        if self.default_system_mode and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": "DEFAULT_SYS"}] + msgs

        # Quirk 2: drop <think>...</think> from assistant messages that aren't last.
        if self.qwq_mode:
            for i in range(len(msgs)):
                m = msgs[i]
                if m.get("role") != "assistant":
                    continue
                # Keep think tokens only on the LAST assistant message overall;
                # strip them from earlier assistant messages.
                is_last_asst = all(m2.get("role") != "assistant" for m2 in msgs[i + 1 :])
                if not is_last_asst:
                    msgs[i] = {
                        "role": "assistant",
                        "content": re.sub(r"<think>.*?</think>", "", m["content"], flags=re.DOTALL),
                    }

        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>{m['content']}<|/{m['role']}|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "".join(parts)

        if tokenize:
            return self.encode(text)
        return text
