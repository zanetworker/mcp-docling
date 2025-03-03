import anyio
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
import io
import os
import logging
import hashlib
import json
import gc
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from docling.document_converter import DocumentConverter
try:
    from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrEngine, EasyOcrOptions
    from docling.datamodel.base_models import InputFormat
except ImportError:
    PdfPipelineOptions = None
    OcrEngine = None
    EasyOcrOptions = None
    InputFormat = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a cache directory
CACHE_DIR = Path.home() / ".cache" / "mcp-docling"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(source: str, enable_ocr: bool, ocr_language: Optional[List[str]]) -> str:
    """Generate a cache key for the document conversion."""
    key_data = {
        "source": source,
        "enable_ocr": enable_ocr,
        "ocr_language": ocr_language or []
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def cleanup_memory():
    """Force garbage collection to free up memory."""
    gc.collect()
    logger.debug("Performed memory cleanup")

def configure_accelerator():
    """Configure the accelerator device for Docling."""
    try:
        # Check if the accelerator_device attribute exists
        if hasattr(settings.perf, 'accelerator_device'):
            # Try to use MPS (Metal Performance Shaders) on macOS
            settings.perf.accelerator_device = AcceleratorDevice.MPS
            logger.info(f"Configured accelerator device: {settings.perf.accelerator_device}")
        else:
            logger.info("Accelerator device configuration not supported in this version of Docling")
        
        # Optimize batch processing
        settings.perf.doc_batch_size = 1  # Process one document at a time
        logger.info(f"Configured batch size: {settings.perf.doc_batch_size}")
        
        return True
    except Exception as e:
        logger.warning(f"Failed to configure accelerator: {e}")
        return False

async def convert_document_impl(
    source: str, 
    enable_ocr: bool = False,
    ocr_language: Optional[List[str]] = None
) -> str:
    try:
        # Remove any quotes from the source string
        source = source.strip('"\'')
        
        # Log the cleaned source
        logger.info(f"Processing document from source: {source}")
        
        # Generate cache key
        cache_key = get_cache_key(source, enable_ocr, ocr_language)
        cache_file = CACHE_DIR / f"{cache_key}.md"
        
        # Check if result is cached
        if cache_file.exists():
            logger.info(f"Using cached result for {source}")
            return cache_file.read_text()
            
        # Log the start of processing
        logger.info(f"Starting conversion of document: {source}")
        
        # Create converter with simple configuration
        converter = DocumentConverter()
        
        # Convert the document
        result = converter.convert(source)
        
        # Export to markdown
        markdown_output = result.document.export_to_markdown()
        
        # Cache the result
        cache_file.write_text(markdown_output)
        logger.info(f"Successfully converted document: {source}")
        
        # Clean up memory to free up resources
        cleanup_memory()
        
        return markdown_output
        
    except Exception as e:
        logger.exception(f"Error converting document: {source}")
        return f"Error converting document: {str(e)}"

async def convert_document_with_images_impl(
    source: str, 
    enable_ocr: bool = False,
    ocr_language: Optional[List[str]] = None
) -> Dict[str, Any]:
    try:
        # Remove any quotes from the source string
        source = source.strip('"\'')
        
        # Configure OCR if enabled
        format_options = {}
        if enable_ocr:
            ocr_options = EasyOcrOptions(lang=ocr_language or ["en"])
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
            format_options = {
                "pdf": {"pipeline_options": pipeline_options}
            }
        
        # Create converter and convert document
        converter = DocumentConverter(format_options=format_options)
        result = converter.convert(source)
        
        # Check for errors - handle different API versions
        has_error = False
        error_message = ""
        
        # Try different ways to check for errors based on the API version
        if hasattr(result, 'status'):
            if hasattr(result.status, 'is_error'):
                has_error = result.status.is_error
            elif hasattr(result.status, 'error'):
                has_error = result.status.error
            
        if hasattr(result, 'errors') and result.errors:
            has_error = True
            error_message = str(result.errors)
        
        if has_error:
            error_msg = f"Conversion failed: {error_message}"
            raise ValueError(error_msg)
            
        # Export to markdown
        markdown_output = result.document.export_to_markdown()
        
        # Extract images
        images = []
        for item in result.document.items:
            if hasattr(item, 'get_image') and callable(getattr(item, 'get_image')):
                try:
                    img = item.get_image(result.document)
                    if img:
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append({
                            "id": item.id,
                            "data": img_str,
                            "format": "png"
                        })
                except Exception:
                    # Skip images that can't be processed
                    pass
        
        return {
            "markdown": markdown_output,
            "images": images
        }
        
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")

async def extract_tables_impl(source: str) -> List[str]:
    source = source.strip('"\'')

    # Create converter and convert document
    converter = DocumentConverter()

    conversion_result = converter.convert(source=source)
    tables_results = []
    for table in conversion_result.document.tables:
        tables_results.append(table.export_to_markdown())

    return tables_results

async def convert_batch_impl(
    sources: List[str],
    enable_ocr: bool = False,
    ocr_language: Optional[List[str]] = None
) -> Dict[str, str]:
    try:
        format_options = {}
        if enable_ocr:
            ocr_options = EasyOcrOptions(lang=ocr_language or ["en"])
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
            format_options = {
                "pdf": {"pipeline_options": pipeline_options}
            }
        
        # Create converter
        converter = DocumentConverter(format_options=format_options)
        
        # Process each document
        results = {}
        for source in sources:
            # Remove any quotes from the source string
            source = source.strip('"\'')
            logger.info(f"Processing document from source: {source}")
            
            try:
                result = converter.convert(source)
                
                # Check for errors - handle different API versions
                has_error = False
                error_message = ""
                
                # Try different ways to check for errors based on the API version
                if hasattr(result, 'status'):
                    if hasattr(result.status, 'is_error'):
                        has_error = result.status.is_error
                    elif hasattr(result.status, 'error'):
                        has_error = result.status.error
                    
                if hasattr(result, 'errors') and result.errors:
                    has_error = True
                    error_message = str(result.errors)
                
                if has_error:
                    results[source] = f"Error: {error_message}"
                else:
                    results[source] = result.document.export_to_markdown()
            except Exception as e:
                results[source] = f"Error: {str(e)}"
        
        return results
        
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")

async def get_system_info_impl() -> Dict[str, Any]:
    try:
        system_info = {
            "batch_settings": {
                "doc_batch_size": settings.perf.doc_batch_size,
                "doc_batch_concurrency": settings.perf.doc_batch_concurrency
            },
            "cache": {
                "enabled": True,
                "location": str(CACHE_DIR)
            }
        }
        
        # Add accelerator info if available
        if hasattr(settings.perf, 'accelerator_device'):
            system_info["accelerator"] = {
                "configured": str(settings.perf.accelerator_device),
                "available": ["CPU", "MPS"]  # Hardcode the common options
            }
        else:
            system_info["accelerator"] = {
                "configured": "Not configured",
                "available": ["CPU"]  # Default to CPU only
            }
            
        return system_info
    except Exception as e:
        raise ValueError(f"Error getting system info: {str(e)}")

def main():
    # Configure accelerator
    configure_accelerator()
    
    app = Server("docling-processor")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        try:
            if name == "convert_document":
                result = await convert_document_impl(
                    source=arguments.get("source", ""),
                    enable_ocr=arguments.get("enable_ocr", False),
                    ocr_language=arguments.get("ocr_language", None)
                )
                return [types.TextContent(type="text", text=result)]
            
            elif name == "convert_document_with_images":
                result = await convert_document_with_images_impl(
                    source=arguments.get("source", ""),
                    enable_ocr=arguments.get("enable_ocr", False),
                    ocr_language=arguments.get("ocr_language", None)
                )
                # Return markdown text and images
                content_items = [types.TextContent(type="text", text=result["markdown"])]
                
                # Add images as embedded resources
                for img in result["images"]:
                    content_items.append(
                        types.ImageContent(
                            type="image",
                            format=img["format"],
                            data=img["data"]
                        )
                    )
                return content_items
            
            elif name == "extract_tables":
                tables = await extract_tables_impl(source=arguments.get("source", ""))
                return [types.TextContent(type="text", text="\n\n".join(tables))]
            
            elif name == "convert_batch":
                result = await convert_batch_impl(
                    sources=arguments.get("sources", []),
                    enable_ocr=arguments.get("enable_ocr", False),
                    ocr_language=arguments.get("ocr_language", None)
                )
                # Format the result as a string
                formatted_result = "\n\n".join([f"## {source}\n\n{content}" for source, content in result.items()])
                return [types.TextContent(type="text", text=formatted_result)]
            
            elif name == "get_system_info":
                result = await get_system_info_impl()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.exception(f"Error in tool call: {name}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="convert_document",
                description="Convert a document from a URL or local path to markdown format",
                inputSchema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or local file path to the document",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for scanned documents",
                            "default": False
                        },
                        "ocr_language": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of language codes for OCR (e.g. [\"en\", \"fr\"])",
                        }
                    },
                },
            ),
            types.Tool(
                name="convert_document_with_images",
                description="Convert a document from a URL or local path to markdown format and return embedded images",
                inputSchema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or local file path to the document",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for scanned documents",
                            "default": False
                        },
                        "ocr_language": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of language codes for OCR (e.g. [\"en\", \"fr\"])",
                        }
                    },
                },
            ),
            types.Tool(
                name="extract_tables",
                description="Extract tables from a document and return them as structured data",
                inputSchema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or local file path to the document",
                        }
                    },
                },
            ),
            types.Tool(
                name="convert_batch",
                description="Convert multiple documents in batch mode",
                inputSchema={
                    "type": "object",
                    "required": ["sources"],
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs or file paths to documents",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for scanned documents",
                            "default": False
                        },
                        "ocr_language": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of language codes for OCR (e.g. [\"en\", \"fr\"])",
                        }
                    },
                },
            ),
            types.Tool(
                name="get_system_info",
                description="Get information about the system configuration and acceleration status",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    # Configure accelerator
    configure_accelerator()
    
    app = Server("docling-processor")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        try:
            if name == "convert_document":
                result = await convert_document_impl(
                    source=arguments.get("source", ""),
                    enable_ocr=arguments.get("enable_ocr", False),
                    ocr_language=arguments.get("ocr_language", None)
                )
                return [types.TextContent(type="text", text=result)]
            
            elif name == "convert_document_with_images":
                result = await convert_document_with_images_impl(
                    source=arguments.get("source", ""),
                    enable_ocr=arguments.get("enable_ocr", False),
                    ocr_language=arguments.get("ocr_language", None)
                )
                # Return markdown text and images
                content_items = [types.TextContent(type="text", text=result["markdown"])]
                
                # Add images as embedded resources
                for img in result["images"]:
                    content_items.append(
                        types.ImageContent(
                            type="image",
                            format=img["format"],
                            data=img["data"]
                        )
                    )
                return content_items
            
            elif name == "extract_tables":
                tables = await extract_tables_impl(source=arguments.get("source", ""))
                return [types.TextContent(type="text", text="\n\n".join(tables))]
            
            elif name == "convert_batch":
                result = await convert_batch_impl(
                    sources=arguments.get("sources", []),
                    enable_ocr=arguments.get("enable_ocr", False),
                    ocr_language=arguments.get("ocr_language", None)
                )
                # Format the result as a string
                formatted_result = "\n\n".join([f"## {source}\n\n{content}" for source, content in result.items()])
                return [types.TextContent(type="text", text=formatted_result)]
            
            elif name == "get_system_info":
                result = await get_system_info_impl()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.exception(f"Error in tool call: {name}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="convert_document",
                description="Convert a document from a URL or local path to markdown format",
                inputSchema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or local file path to the document",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for scanned documents",
                            "default": False
                        },
                        "ocr_language": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of language codes for OCR (e.g. [\"en\", \"fr\"])",
                        }
                    },
                },
            ),
            types.Tool(
                name="convert_document_with_images",
                description="Convert a document from a URL or local path to markdown format and return embedded images",
                inputSchema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or local file path to the document",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for scanned documents",
                            "default": False
                        },
                        "ocr_language": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of language codes for OCR (e.g. [\"en\", \"fr\"])",
                        }
                    },
                },
            ),
            types.Tool(
                name="extract_tables",
                description="Extract tables from a document and return them as structured data",
                inputSchema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or local file path to the document",
                        }
                    },
                },
            ),
            types.Tool(
                name="convert_batch",
                description="Convert multiple documents in batch mode",
                inputSchema={
                    "type": "object",
                    "required": ["sources"],
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs or file paths to documents",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for scanned documents",
                            "default": False
                        },
                        "ocr_language": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of language codes for OCR (e.g. [\"en\", \"fr\"])",
                        }
                    },
                },
            ),
            types.Tool(
                name="get_system_info",
                description="Get information about the system configuration and acceleration status",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        import uvicorn

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
        return 0
    else:
        from mcp.server.stdio import stdio_server

        async def run_stdio():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(run_stdio)
        return 0

if __name__ == "__main__":
    main()