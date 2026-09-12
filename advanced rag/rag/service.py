import os

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

from .config import Settings
from .ingestion import read_cached_chunks
from .memory import SummarizingMemory
from .retrieval import HybridRetriever, ScoredDocument, confidence


REFUSAL = "I could not find relevant information in the knowledge base."

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a careful knowledge-base assistant. Answer only from the supplied
context. Conversation memory may resolve references, but is not factual evidence.
Use concise prose or bullets. Cite factual claims with the matching context number,
for example [1]. If context is incomplete, say so. Never invent a citation.""",
        ),
        (
            "human",
            "Conversation memory:\n{history}\n\nRetrieved context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


class AdvancedRAG:
    def __init__(self, settings: Settings | None = None, enable_reranker: bool | None = None):
        self.settings = settings or Settings()
        self.chunks = read_cached_chunks(self.settings.chunks_file)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.vectorstore = Chroma(
            persist_directory=str(self.settings.db_dir),
            embedding_function=self.embeddings,
            collection_name=self.settings.collection_name,
        )
        use_reranker = self.settings.rerank_enabled if enable_reranker is None else enable_reranker
        reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(self.settings.rerank_model)
        self.retriever = HybridRetriever(
            self.vectorstore,
            self.chunks,
            candidate_count=self.settings.initial_candidates,
            final_count=self.settings.final_results,
            reranker=reranker,
        )
        self.llm = self._create_llm()

        self.retrieval_chain = (
            RunnablePassthrough.assign(results=RunnableLambda(self._retrieve_for_chain))
            | RunnablePassthrough.assign(
                confidence=RunnableLambda(lambda payload: confidence(payload["results"]))
            )
        )
        self.generation_chain = (
            RunnableLambda(self._prepare_generation) | ANSWER_PROMPT | self.llm | StrOutputParser()
        ) if self.llm else None

    def _create_llm(self):
        token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        if not token:
            return None
        endpoint = HuggingFaceEndpoint(
            repo_id=self.settings.llm_repo,
            task="text-generation",
            huggingfacehub_api_token=token,
            max_new_tokens=500,
            temperature=0.1,
        )
        return ChatHuggingFace(llm=endpoint)

    def _retrieve_for_chain(self, payload: dict) -> list[ScoredDocument]:
        return self.retriever.retrieve(payload["question"])

    def _prepare_generation(self, payload: dict) -> dict:
        results = payload["results"]
        context = "\n\n".join(
            f"[{index}] {item.document.page_content}"
            for index, item in enumerate(results, start=1)
        )
        return {"question": payload["question"], "history": payload["history"], "context": context}

    @staticmethod
    def _sources(results: list[ScoredDocument]) -> list[dict]:
        sources = []
        for index, item in enumerate(results, start=1):
            metadata = item.document.metadata
            sources.append(
                {
                    "citation": index,
                    "source": metadata.get("source", "Unknown"),
                    "page": int(metadata.get("page", 0)) + 1,
                    "score": round(item.final_score, 3),
                    "preview": item.document.page_content[:240].strip(),
                }
            )
        return sources

    def ask(self, question: str, threshold: float | None = None, memory=None) -> dict:
        threshold = self.settings.confidence_threshold if threshold is None else threshold
        memory = memory or SummarizingMemory()
        retrieved = self.retrieval_chain.invoke({"question": question})
        results = retrieved["results"]
        score = retrieved["confidence"]
        sources = self._sources(results)
        if score < threshold:
            answer = REFUSAL
        elif not self.generation_chain:
            answer = "Set HUGGINGFACEHUB_ACCESS_TOKEN to generate an answer. Retrieval succeeded."
        else:
            answer = self.generation_chain.invoke(
                {"question": question, "history": memory.render(), "results": results}
            )
            memory.add(question, answer, self.llm)
        return {"answer": answer, "confidence": score, "sources": sources}

    def inspect(self, question: str) -> dict:
        results = self.retriever.retrieve(question)
        return {"confidence": confidence(results), "sources": self._sources(results)}
