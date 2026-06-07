"""Smoke test: ping each model with one short turn."""
import os
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter
from camel.messages import BaseMessage

local_counter = OpenAITokenCounter(ModelType.GPT_4O)

opus = ModelFactory.create(
    model_platform=ModelPlatformType.ANTHROPIC,
    model_type="anthropic--claude-4.7-opus",
    url="http://localhost:3030",
    api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy"),
    model_config_dict={"max_tokens": 100},
    token_counter=local_counter,
)

gpt = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="gpt-5.5",
    url="http://localhost:3030/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    model_config_dict={"max_completion_tokens": 200},
    token_counter=local_counter,
)

for name, model in [("Opus", opus), ("GPT-5.5", gpt)]:
    agent = ChatAgent(system_message="You are a test bot. Reply in one sentence.", model=model)
    msg = BaseMessage.make_user_message(role_name="user", content="Say hi and your model name.")
    try:
        resp = agent.step(msg)
        print(f"[{name}] OK: {resp.msgs[0].content}")
    except Exception as e:
        print(f"[{name}] FAIL: {type(e).__name__}: {e}")
