# Ontotext's GraphDB TTYG & RAGAs Experiments

A collection of practical experiments, evaluation scripts and BPMN resources for using GraphDB's **[Talk To Your Graph (TTYG)](https://graphdb.ontotext.com/documentation/11.0/talk-to-graph.html)** with RAGAs (Retrieval Augmented Generation Assessment) metrics.

This repository provides everything needed to:
- evaluate LLM (Large Language Models) answers to BPMN process queries using both SPARQL (ontology-driven) and ChatGPT Retrieval Connector (context-driven) methods.
- benchmark retrieval performance using RAGAs metrics.
- reproduce the integration setup of GraphDB, Weaviate and the ChatGPT Retrieval Plugin.

---

## Quick Start

**Before running any scripts or experiments, please carefully read and follow the setup steps in:**

> `instructions-to-do-TTYG.txt`

This file details all system requirements, software dependencies, GraphDB and connector configuration, as well as API key management.

---

## Contents

- `RAGAs_TTYG-SPARQL-method-results.py`  
  Run RAGAs evaluation for LLM generated answers grounded in ontology (SPARQL) context.

- `RAGAs_TTYG-ChatGPTRetrievalConnector-method-results.py`  
  Run RAGAs evaluation for LLM generated answers using retrieved text context (ChatGPT Retrieval Connector).

- `BPMN_process-data.ttl`  
  BPMN process serialized as RDF/Turtle.

- `BPMN models/`  
  BPMN diagrams: .png and .adl (Bee-Up export).

- `Average RAGAs scores charts/`  
  Visualization of average RAGAs metric results: bar charts and tabular data.

- `instructions-to-do-TTYG.txt`  
  **Read this first!** Full setup and configuration guide.

---

## Requirements

- Windows 11, 64-bit
- [GraphDB 11.0](https://graphdb.ontotext.com)
- Docker Desktop 4.27.1+
- Python 3.12.1+
- OpenAI API key
- All other dependencies and setup details are in `instructions-to-do-TTYG.txt`.
