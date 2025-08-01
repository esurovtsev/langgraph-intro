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
   - [LangGraph Intro Optimize Chatbot Messages with Memory and Schema Building Blocks for AI Agents](https://www.youtube.com/watch?v=rNYV_Qf9MLk)

6. **Chatbot Summarization** (`06_chatbot-summarization.ipynb`)
   - Advanced memory optimization techniques
   - Implementing conversation summarization
   - Dynamic context management
   - Memory compression strategies
   - [LangGraph Intro - Optimize Chatbot Memory with Summarization: Smarter AI Agents with LangGraph](https://www.youtube.com/watch?v=W0gzRP_g6H0)

7. **Chatbot Persistence** (`07_chatbot-persistence.ipynb`)
   - Implementing MongoDB for long-term memory storage
   - LangGraph Studio integration for workflow visualization
   - Persistent state management across conversations
   - Database configuration and connection handling
   - [LangGraph Intro Persist AI Agent Memory with MongoDB and LangGraph Studio](https://www.youtube.com/watch?v=OjGsw5IeR-g)

8. **Streaming & API Integration** (`08_streaming.ipynb`)
   - LangGraph Studio as local server and API gateway
   - Real-time state streaming techniques (updates/values/messages)
   - API integration patterns for existing workflows
   - Visualizing node-to-node state transitions
   - Hybrid execution (notebook vs studio environments)
   - Preparing for human-in-the-loop workflows
   - [LangGraph Intro Streaming AI Agent State and API Calls with LangGraph Studio](https://www.youtube.com/watch?v=hMHyPtwruVs)

9. **Human-in-the-Loop Workflows** (`09_breakpoints.ipynb`)
   - Implementing execution breakpoints for user approval
   - State inspection and modification during pauses
   - Flow resumption using null parameters
   - Studio integration for visual breakpoint management
   - Infinite tool call support with approval checks
   - Hybrid execution patterns (CLI vs Studio)
   - API parameter handling for flow continuation
   - [LangGraph Intro - Human in the Loop: Breaking and Resuming AI Agent Execution with LangGraph](https://www.youtube.com/watch?v=f0HEm9nY4ec)

10. **Advanced Human Feedback** (`10_human-feedback.ipynb`)
    - State manipulation after breakpoints
    - Interactive feedback collection and processing
    - Message content replacement strategies
    - User feedback node patterns
    - Graph state modification techniques
    - [LangGraph Intro - Human-in-the-Loop: Collecting and Processing User Feedback in AI Agent Workflows](https://www.youtube.com/watch?v=DZVtftZsr60)

11. **Dynamic Breakpoints** (`11_dynamic-breakpoints.ipynb`)
    - Implementing conditional execution breaks
    - Intent validation and flow control
    - Dynamic state inspection during breaks
    - Message content validation and correction
    - Graph resumption with modified state
    - Exception-based breakpoint triggers
    - Studio and API integration for dynamic control
    - [LangGraph Intro - Human-in-the-Loop: Dynamic Breakpoints for AI Agent Control with LangGraph](https://www.youtube.com/watch?v=LvkzIS3OV9w)

12. **State Replay and Forking** (`12_replay-fork-state.ipynb`)
    - Understanding checkpoint-based state management
    - Accessing thread state history and snapshots
    - State replay from historical checkpoints
    - Graph forking and execution branching
    - Thread-based conversation management
    - SDK and API implementation patterns
    - Replay (event playback) vs Fork (real execution)
    - [LangGraph Intro - Human-in-the-Loop: Replaying and Forking AI Agent State with LangGraph](https://www.youtube.com/watch?v=q70GGEHNWgw)

13. **Parallel AI Execution** (`13_parallel-ai-execution.ipynb`)
    - Implementing concurrent node execution
    - State conflict resolution with reducers
    - Super step transaction management
    - Synchronizing parallel execution flows
    - Conditional branching in parallel workflows
    - Building multi-source AI assistants
    - Performance optimization techniques
    - [LangGraph Intro - Running AI Agent Tasks in Parallel with LangGraph](https://www.youtube.com/watch?v=2eMkNLXAs68)

14. **AI Agent Subgraphs** (`14_ai-agent-subgraphs.ipynb`)
    - Implementing subgraphs as modular components
    - Registering subgraphs as nodes in parent graphs
    - Query classification and optimization techniques
    - Multi-source information retrieval (web search and Wikipedia)
    - State synchronization between subgraphs
    - Structured response generation with source attribution
    - Complex workflow orchestration with specialized subgraphs
    - [LangGraph Intro - Structuring AI Agent Workflows with Subgraphs in LangGraph](https://www.youtube.com/watch?v=52oj4SPUHRA)

15. **MapReduce for Parallel Processing** (`15_mapreduce-financial-agent.ipynb`)
    - Implementing the MapReduce pattern for AI agent tasks
    - Dynamic node generation during runtime execution
    - Parallel data processing with mapped nodes
    - State aggregation with reducer functions
    - Building a financial advisor agent with stock analysis
    - Yahoo Finance integration for real-time data
    - Structured output generation with rankings and recommendations
    - Visualizing dynamic parallel workflows in LangGraph Studio
    - [LangGraph Intro – Scaling AI Agents with MapReduce in LangGraph](https://www.youtube.com/watch?v=QI1lvV0rqig)

16. **Multi-Agent Research Pipeline** (`16_multi-agent-research.ipynb`)
    - Building complex multi-agent research workflows
    - Team formation with human-in-the-loop feedback
    - Structured output for analyst profile generation
    - Dialogue-based expert interview subgraphs
    - Parallel execution using map-reduce pattern
    - External search integration (web and Wikipedia)
    - Synthesizing research outputs into comprehensive reports
    - Coordinated autonomous multi-agent system design
    - [LangGraph Intro - Multi Agent Research Pipelines and Report Writing with LangGraph](https://www.youtube.com/watch?v=rCWpJlZdH0c)

17. **Long-Term Memory Store** (`17_longterm-memory-store.ipynb`)
    - Introducing persistent long-term memory in LangGraph
    - Differences between short-term (thread) and long-term (user) memory
    - Organizing memory with namespaces, keys, and values
    - Storing and retrieving user facts and preferences
    - Building chatbots with both short- and long-term memory
    - Personalizing replies using persistent user context
    - Updating and recalling memory across threads and sessions
    - Demonstration in both code and LangGraph Studio
    - [LangGraph Intro – Storing and Using Long-Term Memory in AI Agents with LangGraph](https://www.youtube.com/watch?v=iVtVsI4UTfo)

18. **User Profile Management** (`18_user_profiles.ipynb`)
    - Structuring and managing user profiles with long-term memory
    - Recap of basic user fact storage and its limitations
    - Designing a structured User Profile schema using Pydantic
    - Organizing user attributes: name, profession, seniority, programming languages, frameworks, projects, skills, and interests
    - Integrating structured LLM outputs with LangGraph's memory system
    - Automatically extracting and updating user data during conversations
    - Personalizing chatbot responses based on user profile information
    - Dynamically adding new interests and updating user context over time
    - Demonstration of persistent, structured memory for smarter AI agents
    - [LangGraph Intro – Building and Updating User Profiles for AI Agents with LangGraph](https://www.youtube.com/watch?v=49FjYCpbpQU)

19. **Trustcall for Incremental Memory Extraction** (`19_trustcall-memory-extraction.ipynb`)
    - Introducing Trustcall, a LangChain-native library for robust memory extraction
    - Addressing limitations of full-structure LLM outputs and fragile parsing
    - Enabling partial, incremental updates to user profiles (JSON patching)
    - Configuring Trustcall to extract and patch user profile data from conversations
    - Integrating Trustcall with LangGraph agents for seamless memory updates
    - Demonstrating accurate, consistent, and scalable long-term personalization
    - Hands-on chatbot example with persistent and evolving user profiles
    - [LangGraph Intro – Enhancing AI Agent Memory Extraction with Trustcall in LangGraph](https://www.youtube.com/watch?v=aadbkSuiaNU)

20. **Managing Memory Collections** (`20_memory-collections.ipynb`)
    - Handling collections in long-term memory with Trustcall and LangGraph
    - Defining schemas for memory items (e.g., directives lists)
    - Extracting and updating multiple entries from conversations
    - Storing items with unique keys and supporting partial updates
    - Configuring Trustcall for advanced schema and collection handling
    - Integrating collections into chatbot workflows for dynamic responses
    - Efficient memory node updates based on recent user messages
    - Live demonstration in LangGraph Studio: creating, updating, and leveraging collections in real-time
    - [LangGraph Intro – Managing Memory Collections in AI Agents with Trustcall and LangGraph](https://www.youtube.com/watch?v=9PqEIXrUPdw)

21. **Inspecting Trustcall Patch Updates & Tool Calls** (`21_trustcall-update-inspect.ipynb`)
    - Deep dive into Trustcall's memory update mechanism and JSON patching
    - Demonstrating creation and modification tracking in long-term memory
    - Leveraging LangSmith for tracing Trustcall's data extraction and patch instructions
    - Implementing custom listeners for introspecting tool executions
    - Parsing patch instructions and summarizing memory changes with listeners
    - Understanding patch logic, schema updates, and agent learning insights
    - [LangGraph Intro – Inspecting Trustcall Patch Updates and Memory Tool Calls in LangGraph](https://www.youtube.com/watch?v=YjVtWcD_6Uo)

22. **Developer Mentor AI Agent** (`22_devmentor-agent.ipynb`)
    - Building a Developer Mentor AI Agent using all long-term memory techniques
    - Integrating structured user profiles, instruction lists, and ADRs for context-rich assistance
    - Configuring memory routing and updates with tool calls and Trustcall extractors
    - Structuring and managing multiple memory types for real-world agent use
    - Handling conflicting instructions and leveraging saved architectural decisions
    - Demonstrating chatbot conversations with persistent, evolving preferences and decisions
    - Creating a reusable template for autonomous developer assistants
    - [LangGraph Intro – Building Developer Mentor AI Agent with Long-Term Memory and Structured Updates](https://www.youtube.com/watch?v=PTRHQtsdvDw)

23. **MCP Tooling Integration** (`23_mcp-tooling.ipynb`)
    - Connecting external MCP servers to LangGraph agents for dynamic tooling
    - Transitioning from hardcoded tools to flexible MCP-compatible configurations
    - Loading tools dynamically using MCP adapter and config files
    - Implementing GitHub API integration for real-time repository data
    - Comparing traditional static tool implementation with dynamic MCP approach
    - Building agents that can discover and use tools without manual definition
    - Demonstrating complete workflow from config setup to agent execution
    - [LangGraph Intro – Using MCP Servers to Extend AI Agents with Dynamic Tooling in LangGraph](https://www.youtube.com/watch?v=T9t9kDXY92U)


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
- `langchain_community` - Community integrations for LangChain
- `python-dotenv` - Environment variable management
- `pymongo` - MongoDB driver for persistent storage
- `langgraph-checkpoint-mongodb` - MongoDB checkpoint storage for LangGraph
- `trustcall` - Incremental and robust memory extraction for AI agents
- `yfinance` - Financial data integration (for examples)
- `typing_extensions` - Type hints and annotations
- `wikipedia` - Python library for accessing Wikipedia content

## Environment Setup

1. Create `.env` files in these locations:
   - Root directory: 
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   - Studio directory (if using LangGraph Studio):
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
2. Add your API keys to both locations

## Running Agents in `studio` Using LangGraph Studio (Web Interface)

To run agents (such as those found in the `studio` directory) using the LangGraph Studio web interface for local development, follow these steps:

1. **Install Required Dependencies**
   Make sure all dependencies for your agent are installed:
   ```bash
   pip install -r studio/requirements.txt
   ```

2. **Install the LangGraph CLI**
   ```bash
   pip install -U "langgraph-cli[inmem]"
   ```

4. **Start the Local LangGraph Development Server**
   From the `studio` directory, run:
   ```bash
   langgraph dev
   ```
   This will start the local LangGraph server in watch mode.

4. **Open LangGraph Studio in Your Browser**
   Once the server is running, you can access the Studio UI at:
   [https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)
   (If your server is running at a different host/port, update the `baseUrl` accordingly.)

For more details and troubleshooting, see the [official LangGraph Studio Quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/).


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
