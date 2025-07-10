from typing import Dict, List

from pydantic import BaseModel


class Server(BaseModel):
    type: str
    command: str | None = None
    args: List[str] | None = None
    env: Dict[str, str] | None = None
    url: str | None = None


class Mcp(BaseModel):
    servers: Dict[str, Server]


class Config(BaseModel):
    model: str = 'deepseek-r1'
    mcp: Mcp | None = None


def load_config(config_path: str) -> Config:
    import json
    with open(config_path, 'r') as f:
        data = json.load(f)
        return Config(**data)
