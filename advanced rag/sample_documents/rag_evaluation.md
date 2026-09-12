# Evaluating a RAG Application

RAG evaluation should examine retrieval and generation separately. Retrieval metrics reveal whether useful evidence reached the model. Hit rate measures the fraction of questions for which at least one relevant document appears in the top k results. Recall at k measures how much of the expected relevant material appears. Mean reciprocal rank rewards systems that place the first relevant result nearer the top.

Generation evaluation checks whether the answer is correct, relevant, and supported by retrieved evidence. Faithfulness measures whether claims can be justified by the supplied context. Answer relevance measures whether the response addresses the question. Citation correctness checks whether cited passages actually support their associated claims.

A good evaluation set contains normal questions, paraphrases, ambiguous follow-ups, and unanswerable questions. Unanswerable examples are required to measure false-answer or hallucination rate. Evaluation data should be kept separate from prompt examples and tuning decisions when possible.

Automated metrics make repeated experiments fast, but human review remains useful for nuanced correctness and citation quality. Teams should log the query, retrieved chunks and scores, final answer, latency, and model configuration so failures can be traced to retrieval, reranking, thresholding, or generation.

