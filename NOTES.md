### Design Decisions
- Search space is whole internet because high search scores in small environments do not necessarily  corellate to high search scores in large environments
- Specific known URLs as ground truth to start with objective, testable goal
- Start with retrieval results only and not LLM output to directly test search API performance without masking/bottlenecking with LLM summarization/generation
