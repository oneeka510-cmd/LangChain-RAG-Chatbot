# Responsible Knowledge Assistants

A knowledge assistant should distinguish between retrieved evidence and conversational memory. Memory helps interpret phrases such as “that method” or “its limitations,” but previous assistant statements are not authoritative evidence. Factual claims should remain grounded in source documents.

Source citations improve traceability only when the cited passage supports the claim. Systems should expose document names and locations, preserve metadata through chunking, and avoid displaying a citation when retrieval confidence is too low.

Sensitive or private documents require access controls outside the language model. Production systems should authenticate users, authorize document access before retrieval, encrypt data in transit and at rest, and avoid logging confidential text unnecessarily. Prompt instructions alone are not an access-control mechanism.

Monitoring should include refusal behavior and adversarial or irrelevant questions. A safe failure is an explicit statement that the knowledge base does not contain enough relevant information. This is preferable to an answer assembled from weakly related passages.

