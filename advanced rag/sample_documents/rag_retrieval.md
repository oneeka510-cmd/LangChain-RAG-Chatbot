# Retrieval for RAG Systems

Retrieval-Augmented Generation (RAG) first selects evidence from a knowledge base and then asks a language model to answer from that evidence. Retrieval quality limits answer quality: a generator cannot cite facts that were never retrieved.

Dense vector search embeds queries and document chunks into a shared numerical space. It is useful for semantic similarity and paraphrases. Keyword search such as BM25 rewards exact term overlap and is especially useful for product names, acronyms, identifiers, and rare phrases. Hybrid search combines dense and keyword signals, giving the system both semantic recall and lexical precision.

After candidate retrieval, a reranker evaluates the query and each candidate together. A cross-encoder is more computationally expensive than embedding similarity, but can make a finer relevance judgment. A practical pipeline retrieves a moderately large candidate set cheaply, reranks it, and sends only the best few chunks to the language model.

Chunk size changes the tradeoff between context and precision. Tiny chunks can lose necessary context, while very large chunks may bury the relevant sentence. Chunk overlap helps preserve ideas spanning boundaries but creates duplicates. Metadata such as source, page, and section enables citations and filtering.

Confidence gating prevents weak evidence from becoming a confident-looking answer. A calibrated threshold should be selected using representative evaluation questions rather than intuition alone. When all retrieved evidence is below the threshold, the system should clearly refuse to answer from the knowledge base.

