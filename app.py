from openai import AsyncOpenAI
import chainlit as cl
client = AsyncOpenAI()

cl.instrument_openai()

settings = {
    "model": "gpt-4o",
    "temperature" : 0.3
}

@cl.on_message
async def on_message(message: cl.Message):
    response = await client.chat.completions.create(
        messages=[
            {
                "content": "You are a helpful assistant that can answer questions about the questions asked.",
                "role": "system"
            },
            {
                "content": message.content,
                "role": "user"
            }
        ],
        **settings
    )
    await cl.Message(content=response.choices[0].message.content).send()