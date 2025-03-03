# MCP Docling Server

An MCP server that provides document processing capabilities using the Docling library.

## Installation

You can install the package using pip:

```bash
pip install -e .
```

## Usage

Start the server using either stdio (default) or SSE transport:

```bash
# Using stdio transport (default)
mcp-server-lls

# Using SSE transport on custom port
mcp-server-lls --transport sse --port 8000
```

If you're using uv, you can run the server directly without installing:

```bash
# Using stdio transport (default)
uv run mcp-server-lls

# Using SSE transport on custom port
uv run mcp-server-lls --transport sse --port 8000
```

## Available Tools

The server exposes the following tools:

1. **convert_document**: Convert a document from a URL or local path to markdown format
   - `source`: URL or local file path to the document (required)
   - `enable_ocr`: Whether to enable OCR for scanned documents (optional, default: false)
   - `ocr_language`: List of language codes for OCR, e.g. ["en", "fr"] (optional)

2. **convert_document_with_images**: Convert a document and extract embedded images
   - `source`: URL or local file path to the document (required)
   - `enable_ocr`: Whether to enable OCR for scanned documents (optional, default: false)
   - `ocr_language`: List of language codes for OCR (optional)

3. **extract_tables**: Extract tables from a document as structured data
   - `source`: URL or local file path to the document (required)

4. **convert_batch**: Process multiple documents in batch mode
   - `sources`: List of URLs or file paths to documents (required)
   - `enable_ocr`: Whether to enable OCR for scanned documents (optional, default: false)
   - `ocr_language`: List of language codes for OCR (optional)

5. **get_system_info**: Get information about system configuration and acceleration status

## Example with Llama Stack


https://github.com/user-attachments/assets/8ad34e50-cbf7-4ec8-aedd-71c42a5de0a1


You can use this server with [Llama Stack](https://github.com/meta-llama/llama-stack) to provide document processing capabilities to your LLM applications. Make sure you have a running Llama Stack server, then configure your `INFERENCE_MODEL`

```python
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.shared_params.url import URL
from llama_stack_client import LlamaStackClient
import os

# Set your model ID
model_id = os.environ["INFERENCE_MODEL"]
client = LlamaStackClient(
    base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
)

# Register MCP tools
client.toolgroups.register(
    toolgroup_id="mcp::docling",
    provider_id="model-context-protocol",
    mcp_endpoint=URL(uri="http://0.0.0.0:8000/sse"))

# Define an agent with MCP toolgroup
agent_config = AgentConfig(
    model=model_id,
    instructions="""You are a helpful assistant with access to tools that can convert documents to markdown.
When asked to convert a document, use the 'convert_document' tool.
You can also extract tables with 'extract_tables' or get images with 'convert_document_with_images'.
Always use the appropriate tool when asked to process documents.""",
    toolgroups=["mcp::docling"],
    tool_choice="auto",
    max_tool_calls=3,
)

# Create the agent
agent = Agent(client, agent_config)

# Create a session
session_id = agent.create_session("test-session")

# Define the prompt
prompt = "Please convert the document at https://arxiv.org/pdf/2004.07606 to markdown and summarize its content."

# Create a turn
response = agent.create_turn(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    session_id=session_id,
)

# Log the response
for log in EventLogger().log(response):
    log.print()
```

## Caching

The server caches processed documents in `~/.cache/mcp-docling/` to improve performance for repeated requests.

