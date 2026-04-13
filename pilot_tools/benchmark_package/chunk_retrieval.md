# Passage Chunk Retrieval Protocol

## Motivation

Evidence windows extracted after full-document retrieval are currently a likely clean-path bottleneck.
The benchmark therefore defines a chunk-level retrieval path as the next clean ceiling experiment.

## Recommended chunk schema

- chunk size: `150-250` tokens
- overlap: `40-80` tokens
- keep title in every chunk
- preserve `parent_docid`
- preserve `chunk_idx`

Chunk record format:

```json
{
  "id": "doc123::chunk::7",
  "contents": "Title\nchunk text ...",
  "meta": {
    "parent_docid": "doc123",
    "title": "Title",
    "chunk_idx": 7,
    "token_start": 1050,
    "token_end": 1250,
    "chunk_size": 200,
    "overlap": 50
  }
}
```

## Tooling

- Build chunk corpus: `python3 build_chunk_corpus.py ...`
- Build chunk Lucene index: `python3 pyserini_index.py ...`
- Build chunk FAISS index: `python3 build_faiss_index.py ...`

## Experimental order

1. NQ chunk retrieval + vote reader
2. Hotpot chunk retrieval + rerank + multi-doc single reader
3. Hotpot chunk retrieval + 2-hop if the chunk index remains stable
4. Combined dataset-aware clean base

## Success criteria

- NQ should match or exceed the current `doc[:1024]` base
- Hotpot should recover more than evidence-window or vote-style readers
- Combined Base should exceed the current `34-35%` plateau

## Risk notes

- Chunk retrieval can break 2-hop query quality if chunks are too narrow
- Overlap that is too small may drop bridge facts
- Overlap that is too large may recreate sybil redundancy
