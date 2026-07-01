# Tools


## rag_search

Type:
`retrieval`

Purpose:
Lexical RAG over indexed sources + Project Brain (high priority).

Inputs:

- query
- top_k

Outputs:

- hits with path, snippet, score

Used By:

- workspace assistant modes with RAG

---


## project_brain_tool

Type:
`brain`

Purpose:
Regenerate project_brain from scan (refresh/reindex/scan) or write model Markdown (write_brain: brain_rel_path + content; write_architecture for agent_architecture.md only).

Inputs:

- action
- brain_rel_path (write_brain)
- content
- write_mode

Outputs:

- written paths
- brain_chunks_indexed
- brain_rel_path on write

Used By:

- after structural changes or when supplementing brain docs

---


## read_file / edit_file / write_file

Type:
`filesystem`

Purpose:
Read and mutate workspace files.

Inputs:

- path
- content/lines

Outputs:

- file fragments or confirmations

Used By:

- full edit modes

---

