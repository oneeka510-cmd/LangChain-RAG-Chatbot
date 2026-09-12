from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


SUMMARY_PROMPT = PromptTemplate.from_template(
    """Update the concise conversation summary. Preserve names, decisions, facts, and
unresolved references needed for later questions. Do not add external facts.

Existing summary:
{summary}

New conversation:
{conversation}

Updated summary:"""
)


class SummarizingMemory:
    def __init__(self, max_recent_messages: int = 8) -> None:
        self.max_recent_messages = max_recent_messages
        self.messages: list[HumanMessage | AIMessage] = []
        self.summary = ""

    def add(self, question: str, answer: str, llm=None) -> None:
        self.messages.extend([HumanMessage(content=question), AIMessage(content=answer)])
        if len(self.messages) <= self.max_recent_messages:
            return
        old = self.messages[:2]
        self.messages = self.messages[2:]
        conversation = "\n".join(
            f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: {message.content}"
            for message in old
        )
        if llm:
            self.summary = (SUMMARY_PROMPT | llm | StrOutputParser()).invoke(
                {"summary": self.summary, "conversation": conversation}
            )
        else:
            self.summary = (self.summary + "\n" + conversation)[-2500:]

    def render(self) -> str:
        recent = "\n".join(
            f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: {message.content}"
            for message in self.messages
        )
        return f"Summary: {self.summary or 'None'}\nRecent messages:\n{recent or 'None'}"

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

