# LangGraph Introduction

This repository contains a series of Jupyter notebooks that demonstrate how to work with LangGraph - a powerful framework for building structured agents and workflows using LangChain. The project provides a hands-on approach to learning LangGraph concepts, from basic principles to advanced implementations.

## Overview

The tutorial is structured in a progressive manner, with each notebook building upon concepts introduced in previous ones. Each concept is accompanied by both code examples, making it easier to understand the underlying principles.

## Contents

1. **Chain Basics** (`01_chain.ipynb`)
   - Introduction to basic graph concepts
   - Understanding state management
   - Implementation of simple chains
   - [LangGraph Intro - Learn How to Build AI Agents with Tools and Graphs](https://www.youtube.com/watch?v=8azc-h0INTQ)

2. **Router Implementation** (`02_router.ipynb`)
   - Working with routers in LangGraph
   - Flow control and decision making
   - [LangGraph Intro - Build Router AI Agents with LangGraph Tools and LLMs](https://www.youtube.com/watch?v=MN9RD8sAKjk)

3. **Agent Creation** (`03_agent.ipynb`)
   - Building agents with LangGraph
   - Agent architecture and components
   - [LangGraph Intro - Build Autonomous AI Agents with ReAct and LangGraph Tools](https://www.youtube.com/watch?v=ZfjaIshGkmk)

4. **Agent with Memory** (`04_agent-memory.ipynb`)
   - Implementing memory capabilities
   - State persistence and management
   - Advanced agent patterns
   - [LangGraph Intro Build AI Agents with Memory Using LangGraph and LLMs](https://www.youtube.com/watch?v=pLaXs--af14)

5. **Chatbot Message Management** (`05_chatbot-messages.ipynb`)
   - Managing conversation context and history
   - Token usage optimization
   - Message state handling
   - Efficient message trimming strategies

## Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab environment

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd langgraph-intro
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project relies on the following main packages:
- `langgraph` - Core framework for building graph-based workflows
- `langchain_openai` - LangChain OpenAI integration
- `langchain_core` - Core LangChain functionalities
- `python-dotenv` - Environment variable management
- `yfinance` - Financial data integration (for examples)
- `typing_extensions` - Type hints and annotations

## Environment Setup

1. Create a `.env` file in the root directory
2. Add your required API keys and configurations:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Ensure your virtual environment is activated
2. Launch Jupyter Notebook:
```bash
jupyter notebook
```
3. Navigate through the notebooks in numerical order
4. Each notebook contains detailed explanations and executable code examples

## Video Tutorials

Each notebook has an accompanying video tutorial that walks through the concepts in detail. The video links are provided in each section above.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- LangChain team for providing the foundational frameworks
- OpenAI for API integration capabilities
