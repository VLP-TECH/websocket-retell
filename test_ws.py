import asyncio
import websockets
import json

async def test():
    uri = "wss://bf62-90-162-180-179.ngrok-free.app/llm-websocket/test123"
    async with websockets.connect(uri) as ws:
        print("Conectado!")
        msg = await ws.recv()
        print(f"Recibido: {msg}")

asyncio.run(test())