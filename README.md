# SHL RAG Bot: Technical Approach

## Architecture Overview
The application follows a modern, decoupled architecture with a Next.js frontend and a FastAPI Python backend. The system diagram shows a clear separation between frontend and backend components, with API communication between them.

### Frontend (Next.js)
- **Framework**: Next.js 15.2.4 with React 19
- **UI Components**: Combination of custom components and Radix UI primitives
- **State Management**: React's built-in state management with hooks
- **Styling**: Tailwind CSS 4.1.3 for utility-first styling
- **Key Components**:
  - Chat Interface & Message components
  - Recommendation Table
  - Theme Provider
  - Responsive design with desktop/mobile support

### Backend (Python FastAPI)
- **Framework**: FastAPI for high-performance API endpoints
- **RAG Implementation**: LlamaIndex as the core RAG framework
- **LLM Integration**: Groq integration via `llama-index-llms-groq`
- **Vector Database**: AstraDB for efficient vector storage via `llama-index-vector-stores-astra-db`
- **Embeddings**: NVIDIA embeddings via `llama-index-embeddings-nvidia`
- **Data Processing**:
  - BeautifulSoup4 for web scraping
  - Playwright for browser automation
  - Pandas for data manipulation
  - OpenPyXL for Excel file processing

### Data Flow
1. User sends a query through the chat interface
2. Frontend sends an API request to the backend
3. Backend processing modules:
   - Query the vector database for relevant documents
   - Combine retrieved context with the user query
   - Generate a response using the LLM
   - Send structured response back to frontend
4. Frontend renders the response with appropriate UI components

### Development Workflow
- **Development**: Next.js dev server with Turbopack for fast refreshes
- **Build Process**: Standardized build scripts for both frontend and backend
- **Code Quality**: ESLint for JavaScript/TypeScript linting
- **Dependency Management**: npm for frontend, Python project structure for backend

### Evaluation & Testing
- DeepEval for evaluating RAG system performance
- Continuous testing to ensure response quality and accuracy

## Implementation Highlights
- Modular architecture allowing for easy component replacement
- Separation of concerns between UI, state management, and data retrieval
- Optimized for both development experience and production performance
- Responsive design supporting various device types
- Type safety through TypeScript on the frontend and Pydantic on the backend

![Architecture Diagram](diagram.png)