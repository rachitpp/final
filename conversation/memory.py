from typing import List, Tuple


class ConversationMemory:
    """
    Lightweight in-memory chat history. Bounded ring of recent turns.
    No persistence, no vector store, no framework dependencies.
    """

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self._turns: List[Tuple[str, str]] = []

    def add(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def turns(self) -> List[Tuple[str, str]]:
        """Return a copy of the recent turns."""
        return list(self._turns)

    def is_empty(self) -> bool:
        return not self._turns

    def clear(self) -> None:
        self._turns.clear()
