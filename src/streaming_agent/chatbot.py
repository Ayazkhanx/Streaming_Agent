import chainlit as cl

from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from dotenv import load_dotenv, find_dotenv
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import os

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_KEY")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(name="personal agent", instructions="You are a helpful agent. Provide any info that somebody need but not about abusive and guilty things like porn etc ", model=model )


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am your helpful agent. tell me how can i help you ?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent, 
        input=history, 
        run_config=config)
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
            
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set('history', history)
    await cl.Message(
        content=result.final_output
    )#.send()