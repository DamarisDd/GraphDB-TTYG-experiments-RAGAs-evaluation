# Ontotext's GraphDB TTYG & RAGAs experiments

A collection of practical experiments, evaluation scripts and BPMN resources for using GraphDB's **[Talk To Your Graph (TTYG)](https://graphdb.ontotext.com/documentation/11.0/talk-to-graph.html)** with RAGAs (Retrieval Augmented Generation Assessment) metrics.

This repository provides everything needed to:
- evaluate LLM (Large Language Models) answers to BPMN process queries using both SPARQL (ontology-driven) and ChatGPT Retrieval Connector (context-driven) methods.
- benchmark retrieval performance using RAGAs metrics.
- reproduce the integration setup of GraphDB, Weaviate and the ChatGPT Retrieval Plugin.

---

## Quick start

**Before running any scripts or experiments, please carefully read and follow the setup steps in:**

> `instructions-to-do-TTYG.txt`

This file details all system requirements, software dependencies, GraphDB and connector configuration, as well as API key management.

---

## Contents

- `Average RAGAs scores charts/`  
  Visualization of average RAGAs metric results: bar charts and tabular data.

- `BPMN models/`  
  BPMN diagrams: .png and .adl (Bee-Up export).

- `RAGAs_results_ChatGPTRetrievalConnectormethod/`  
  Evaluation results for the ChatGPT Retrieval Connector method.

- `RAGAs_results_SPARQLmethod/`  
  Evaluation results for the SPARQL/ontology-based method.

- `BPMN_process-data.ttl`  
  BPMN process serialized as RDF/Turtle.

- `RAGAs_TTYG-ChatGPTRetrievalConnector-method-results.py`  
  RAGAs evaluation script for answers using the ChatGPT Retrieval Connector.

- `RAGAs_TTYG-SPARQL-method-results.py`  
  RAGAs evaluation script for answers using ontology-based SPARQL.

- `create-retrieval-bpmnprocess.rq`  
  SPARQL file to create the ChatGPT Retrieval Connector instance in GraphDB.

- `docker-compose.yml`  
  Configuration file to run Weaviate and dependencies with Docker.

- `instructions-to-do-TTYG.txt`  
  **Start here.** Full setup and configuration guide.

- `ontology.txt`  
  Ontology file used as LLM context for the RAGAs evaluations.

- `requirements.txt`  
  Python requirements for the evaluation scripts.

- `retrieved_context_for_LLM.txt`  
  Retrieved context fragments used as LLM context for the RAGAs evaluations.

- `run-poetry-bpmnprocess.sh`  
  Script to launch the ChatGPT Retrieval Plugin with BPMN-specific configuration.

---

## Requirements

- Windows 11, 64-bit
- [GraphDB 11.0](https://graphdb.ontotext.com)
- Docker Desktop 4.27.1+
- Python 3.12.1+
- OpenAI API key
- All other dependencies and setup details are in `instructions-to-do-TTYG.txt`.
