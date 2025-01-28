    A[Main Application] -->|initializes| B[QueryTranslator]
    B -->|uses| C[TextPreprocessor]
    B -->|uses| D[SemanticAnalyzer]
    B -->|uses| E[VectorManager]
    B -->|uses| F[SQLValidator]
    D -->|uses| G[HFEmbeddingModel]
    D -->|uses| H[DomainContext]
    D -->|uses| I[QueryIntent]
    D -->|uses| J[VectorSearchResult]
    B -->|uses| K[LLMAPIClient]
    B -->|uses| L[VectorAPIClient]
    M[ModelCatalog] -->|loads| G
    N[ChromaVectorAPIClient] -->|interacts with| L
    O[OpenAILLMClient] -->|interacts with| K
    P[LLMWareAPIClient] -->|interacts with| K
    Q[SQLValidator] -->|validates| R[SQL Queries]
    S[Domain Terms] -->|used by| B
    S -->|used by| Ds