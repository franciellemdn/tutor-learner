import os
import sys
import contextlib
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool

class MCPClientManager:
    """
    Manages the connection and lifecycle of a local stdio-based MCP server.
    Converts exposed MCP tools into LangChain-compatible tools.
    """
    def __init__(self):
        self.session = None
        self._exit_stack = None

    async def connect(self):
        # Absolute path to search_server.py inside app/mcp/
        server_script = os.path.join(os.path.dirname(__file__), "search_server.py")
        
        # Configure stdio execution parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env=os.environ.copy()
        )
        
        self._exit_stack = contextlib.AsyncExitStack()
        print(f"[MCP CLIENT] Starting MCP server subprocess: {sys.executable} {server_script}...")
        
        try:
            # Enter stdio connection context
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            # Enter client session context
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Initialize the session handshake
            await self.session.initialize()
            print("[MCP CLIENT] Handshake successful, MCP server connected.")
        except Exception as e:
            print(f"[MCP CLIENT Error] Failed to connect to server: {e}")
            await self.disconnect()
            raise

    async def disconnect(self):
        if self._exit_stack:
            print("[MCP CLIENT] Shutting down MCP server process...")
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                print(f"[MCP CLIENT Warning] Exception while closing stack: {e}")
            self._exit_stack = None
            self.session = None
            print("[MCP CLIENT] Subprocess terminated.")

    async def get_langchain_tools(self):
        if not self.session:
            print("[MCP CLIENT Warning] Session not connected. Cannot fetch tools.")
            return []
            
        try:
            print("[MCP CLIENT] Requesting tool listing from server...")
            mcp_tools_result = await self.session.list_tools()
            langchain_tools = []
            
            for mcp_tool in mcp_tools_result.tools:
                print(f"[MCP CLIENT] Wrapping tool: '{mcp_tool.name}' -> '{mcp_tool.description}'")
                
                # Closure helper to correctly capture tool names during iteration
                def make_langchain_tool(name=mcp_tool.name, desc=mcp_tool.description):
                    async def mcp_tool_wrapper(query: str) -> str:
                        """Proxy tool executing queries on the remote MCP Server."""
                        try:
                            # Invoke tool on the server
                            result = await self.session.call_tool(name, arguments={"query": query})
                            
                            # Parse content responses (expecting TextContent blocks)
                            output_texts = []
                            for block in result.content:
                                if hasattr(block, "text"):
                                    output_texts.append(block.text)
                                elif isinstance(block, dict) and "text" in block:
                                    output_texts.append(block["text"])
                                else:
                                    output_texts.append(str(block))
                            return "\n".join(output_texts)
                        except Exception as inner_err:
                            return f"Tool execution failed: {str(inner_err)}"
                            
                    def dummy_sync(query: str) -> str:
                        raise NotImplementedError("This tool is async-only. Use ainvoke.")

                    return StructuredTool.from_function(
                        func=dummy_sync,
                        coroutine=mcp_tool_wrapper,
                        name=name,
                        description=desc
                    )
                
                langchain_tools.append(make_langchain_tool())
                
            return langchain_tools
        except Exception as e:
            print(f"[MCP CLIENT Error] Failed to wrap tools: {e}")
            return []
