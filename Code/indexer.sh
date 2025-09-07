PYTHON -m pyserini.index.lucene \
--input /GS_file \
--collection JsonCollection \
--generator DefaultLuceneDocumentGenerator \
--index /index \
--threads 1 \
--storePositions --storeDocvectors --storeRaw
