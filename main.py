import argparse
import asyncio
import sys
from typing import Dict, List

from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from pydantic import ValidationError

from config import Config, Mcp, load_config

SYSTEM_PROMPT = "你是一个会使用MCP工具，并擅长使用简体中文进行回答问题的AI Agent。请根据用户的请求，结合使用MCP工具来完成任务。"

verbose = False


async def setup_agent(llm: LLM, toolSpecs: List[McpToolSpec]) -> FunctionAgent:
    tools = []
    for toolSpec in toolSpecs:
        tools.extend(await toolSpec.to_tool_list_async())
    return FunctionAgent(
        name="mcp_agent",
        description="一个会使用MCP工具的AI Agent",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )


def setup_llm(model: str = "qwen3") -> LLM:
    from llama_index.llms.ollama import Ollama

    return Ollama(model=model, request_timeout=120.0)


def setup_mcp_tool_specs(config: Mcp | None) -> Dict[str, McpToolSpec]:
    if not config:
        return {}

    tool_specs = {}
    for name, spec in config.servers.items():
        client = None
        if spec.type == "http" and spec.url:
            client = BasicMCPClient(spec.url)
        elif spec.type == "stdio" and spec.command:
            client = BasicMCPClient(spec.command, args=spec.args, env=spec.env)
        if client:
            tool_specs[name] = McpToolSpec(client=client)

    return tool_specs


async def handle_user_input(
    agent: FunctionAgent, agent_ctx: Context, user_input: str, verbose: bool = False
) -> str:
    handler = agent.run(user_input, context=agent_ctx)
    async for event in handler.stream_events():
        if verbose and isinstance(event, ToolCall):
            print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif verbose and isinstance(event, ToolCallResult):
            print(f"Tool {event.tool_name} returned {event.tool_output}")

    response = await handler

    return str(response)


async def main(config: Config):
    llm = setup_llm(config.model)
    mcp_tool_specs = setup_mcp_tool_specs(config.mcp)
    agent = await setup_agent(llm, list(mcp_tool_specs.values()))
    agent_ctx = Context(agent)

    if verbose:
        print("Agent可以使用以下MCP servers：")
        for name, spec in mcp_tool_specs.items():
            tools = await spec.to_tool_list_async()
            print(f" - {name}")
            for tool in tools:
                print(f"  - {tool.metadata.name}: {tool.metadata.description}")

    # 处理用户输入的主循环
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Bye!")
                break
            print(f"用户: {user_input}")
            response = await handle_user_input(agent, agent_ctx, user_input, verbose)
            print(f"AI Agent: {response}")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"FATAL: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AI Agent with MCP tools.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
        help="Path to the config file.",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for tool calls."
    )

    args = parser.parse_args()

    verbose = args.verbose
    config_path = args.config

    try:
        config = load_config(config_path)
    except ValidationError as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading config: {e}")
        sys.exit(1)

    asyncio.run(main(config))
