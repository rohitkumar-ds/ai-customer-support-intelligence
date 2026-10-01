"""Bounded in-process conversation memory.

Only the most recent configurable messages are retained. This avoids blindly sending
unlimited history to an LLM. Production deployments can replace this store with a database.
"""
from collections import defaultdict, deque
class ConversationMemory:
    """Keep recent role/content pairs per conversation ID."""

    def __init__(self,max_messages=6):
        self.max_messages=max_messages; self._data=defaultdict(lambda:deque(maxlen=max_messages))
    def add(self,conversation_id,role,content):
        """Append one message while automatically evicting old history."""
        self._data[conversation_id].append({"role":role,"content":content})
    def get(self,conversation_id):
        """Return a copy of recent conversation history."""
        return list(self._data.get(conversation_id,[]))
