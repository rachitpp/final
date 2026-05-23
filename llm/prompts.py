from langchain_core.prompts import ChatPromptTemplate


# Final answer prompt — only this output reaches the user.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and helpful AI assistant.

Rules:
- Answer ONLY from the provided context.
- Match the depth of your answer to what the user asks for. If they ask for
  a detailed explanation, be thorough and comprehensive — cover problem
  statement, mathematical formulation, intuition, and worked examples where
  the context supports it. If they ask a quick factual question, be brief.
- Structure your response clearly with sections or bullet points where helpful.
- If the answer is not in the context, say:
  "I could not find the answer in the provided documents."
- Do not hallucinate or add information not present in the context.
- When citing a source, use the format (filename, p.N) — for example
  (MachineLearning.pdf, p.29). Never reference internal chunk numbers.
"""),
    ("human", """Context:
{context}

Question: {question}

Answer:"""),
])


# Document summary prompt — one call per PDF at ingestion time.
# Output is embedded and stored to power query routing.
SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You write concise document summaries for a RAG routing system.
Given a sample of text from a document, write 4-6 sentences that capture:
- The main topic and domain of the document
- Key concepts, algorithms, or themes covered
- The type and level of the document (introductory textbook, research paper, technical manual, etc.)
Output only the summary. No preamble."""),
    ("human", """Document: {source}

Content sample:
{content}

Summary:"""),
])


# HYDE prompt — output is used as a semantic search query only,
# NEVER shown to the user.
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You write short, factual hypothetical passages used \
for semantic document retrieval. Write a single dense paragraph that \
would answer the user's question if it appeared in a textbook or \
technical reference. Use domain-specific terminology. Do not refuse, \
do not hedge, do not say you don't know. Output only the paragraph."""),
    ("human", "Question: {question}\n\nPassage:"),
])


# Query-rewrite prompt — turns follow-ups into standalone questions.
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You rewrite follow-up questions into standalone \
questions using the conversation history.

Rules:
- If the question is already standalone, return it unchanged.
- Otherwise, resolve pronouns and implicit references using the history.
- Preserve the user's intent and specificity.
- Output ONLY the rewritten question, nothing else."""),
    ("human", """Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""),
])


# Contextualization prompt — produces a terse retrieval-oriented header
# that gets PREPENDED to each chunk before embedding. Goal: enrich the
# chunk's embedding with its surrounding topic and provenance so isolated
# chunks ("...the convolution of a patch...") still retrieve well.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You write a short retrieval-oriented context header \
for a document chunk. The header will be prepended to the chunk and \
embedded together with it.

Rules:
- Output 1–3 sentences. No preamble, no headings, no quotes.
- State the chunk's topic and where it sits in the document.
- Use the document title and section name if useful.
- Be specific. Prefer concrete terms from the chunk over generic phrasing.
- Do not summarize the chunk verbatim. Do not invent facts."""),
    ("human", """Document: {source}
Section: {section}

Chunk:
{chunk}

Context header:"""),
])
