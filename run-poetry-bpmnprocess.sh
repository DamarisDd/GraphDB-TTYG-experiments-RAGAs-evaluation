#!/bin/sh

# Authentication token to access the plugin, replace with actual value
export BEARER_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

# Your OpenAI API KEY, replace with actual value
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

# Vector database to use, in our case Weaviate
export DATASTORE=weaviate
# Weaviate's URL
export WEAVIATE_URL=http://localhost:8080
# Weaviate's schema name
export WEAVIATE_CLASS=BPMNPROCESS

# Text chunk size for indexing
export CHUNK_SIZE=400

poetry run start
