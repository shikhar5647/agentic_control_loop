# 🏭 PFD Control Loop Prediction System

AI-powered control structure design for Process Flow Diagrams using multi-agent architecture, classical control theory, and chemical engineering principles.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)

## 🎯 Overview

This system uses a **multi-agent architecture** built with LangGraph to predict optimal control loop pairings for chemical processes. It combines:

- **Classical Control Theory**: RGA, SVD, Interaction Analysis
- **Chemical Engineering Principles**: Unit operation heuristics, Luyben's rules
- **AI Reasoning**: Google Gemini for intelligent decision-making
- **Multi-Agent Workflow**: Specialized agents for comprehensive analysis

## 🌟 Features

### Core Capabilities
- ✅ **RGA Analysis** - Relative Gain Array for variable pairing
- ✅ **SVD Controllability** - Singular Value Decomposition assessment
- ✅ **Interaction Minimization** - Loop coupling analysis
- ✅ **Chemical Engineering Heuristics** - Process-specific strategies
- ✅ **Multi-Agent Architecture** - LangGraph-based workflow
- ✅ **Comprehensive Validation** - Safety and performance checks

### Technical Features
- 🤖 5 specialized AI agents working collaboratively
- 📊 Interactive Streamlit web interface
- 📈 Real-time visualization of results
- 📥 Export to JSON, CSV, and Markdown
- 🔄 Supports multiple process types
- ⚡ Efficient workflow orchestration

## 🏗️ Architecture

### Multi-Agent Workflow

```
┌─────────────────────┐
│  PFD Analyzer       │  Analyzes process structure
│  Agent              │  Identifies control requirements
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RGA Calculator     │  Computes Relative Gain Array
│  Agent              │  Recommends pairings
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Controllability    │  Performs SVD analysis
│  Analyzer Agent     │  Assesses system properties
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Pairing Optimizer  │  Synthesizes optimal pairings
│  Agent              │  Integrates all analyses
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Validation Agent   │  Validates control structure
│                     │  Provides recommendations
└─────────────────────┘
```

