import json
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

# --- CONFIGURACIÓN ---
OLLAMA_URL = "https://ia-tools-ollama.rzd02y.easypanel.host"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

QDRANT_URL = "https://ia-tools-qdrant.rzd02y.easypanel.host"
QDRANT_COLLECTION = "avidocumentation2"

OPENING_MESSAGE = "Hola, gracias por llamar a CDV Consulting. Soy Gabriel, ¿en qué puedo ayudarte?"

SYSTEM_PROMPT = """Eres Gabriel, asistente de CDV Consulting. Responde en español, tono natural, profesional y cercano. 
Usa contexto si se proporciona. Si no sabes algo, dilo claramente y ofrece derivar al equipo."""

# --- FUNCIONES RAG ---

async def get_embedding(text: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": OLLAMA_EMBED_MODEL,
                    "prompt": text
                }
            )

        data = response.json()
        print("[EMBED RAW]", data)

        if "embedding" in data:
            return data["embedding"]

        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]

        raise Exception(f"Formato embedding desconocido: {data}")

    except Exception as e:
        print(f"[RAG ERROR - EMBEDDING] {e}")
        return []


async def search_qdrant(query: str, top_k: int = 3) -> str:
    try:
        embedding = await get_embedding(query)

        if not embedding:
            print("[RAG] embedding vacío")
            return ""

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                headers={
                    "api-key": QDRANT_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "vector": embedding,
                    "limit": top_k,
                    "with_payload": True
                }
            )

        data = response.json()
        print("[QDRANT RAW]", data)

        results = data.get("result", [])

        if not results:
            return ""

        contexto = "\n\n".join([
            r.get("payload", {}).get("text")
            or r.get("payload", {}).get("content", "")
            for r in results
        ])

        print(f"[RAG] {len(results)} fragmentos encontrados")
        return contexto

    except Exception as e:
        print(f"[RAG ERROR - QDRANT] {e}")
        return ""


async def call_ollama(messages: list) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 150
                    }
                }
            )

        data = response.json()
        print("[OLLAMA RAW]", data)

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        if "error" in data:
            return f"Error del modelo: {data['error']}"

        return "Disculpa, ha habido un problema generando la respuesta."

    except Exception as e:
        print(f"[ERROR OLLAMA] {e}")
        return "Disculpa, ha habido un problema técnico."


# --- WEBSOCKET HANDLER ---

@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"[OPEN] call_id={call_id}")

    # Mensaje inicial
    await websocket.send_text(json.dumps({
        "response_type": "response",
        "response_id": 0,
        "content": OPENING_MESSAGE,
        "content_complete": True,
        "end_call": False
    }))

    try:
        async for data in websocket.iter_text():
            request = json.loads(data)
            interaction_type = request.get("interaction_type")

            # Ping/pong
            if interaction_type == "ping_pong":
                await websocket.send_text(json.dumps({
                    "response_type": "ping_pong",
                    "timestamp": request["timestamp"]
                }))
                continue

            if interaction_type == "update_only":
                continue

            if interaction_type in ("response_required", "reminder_required"):

                transcript = request.get("transcript", [])

                # Última pregunta del usuario
                ultima_pregunta = ""
                for turn in reversed(transcript):
                    if turn["role"] == "user" and turn.get("content", "").strip():
                        ultima_pregunta = turn["content"].strip()
                        break

                print(f"[USER] {ultima_pregunta}")

                # RAG
                contexto = ""
                if ultima_pregunta:
                    contexto = await search_qdrant(ultima_pregunta)

                # Construcción de mensajes
                system_msg = SYSTEM_PROMPT
                if contexto:
                    system_msg += f"\n\nCONTEXTO:\n{contexto}"

                messages = [{"role": "system", "content": system_msg}]

                for turn in transcript:
                    role = "assistant" if turn["role"] == "agent" else "user"
                    content = turn.get("content", "").strip()
                    if content:
                        messages.append({"role": role, "content": content})

                print(f"[OLLAMA] enviando {len(messages)} mensajes")

                # Llamada al modelo
                reply = await call_ollama(messages)

                print(f"[REPLY] {reply}")

                # Respuesta al cliente
                await websocket.send_text(json.dumps({
                    "response_type": "response",
                    "response_id": request.get("response_id"),
                    "content": reply,
                    "content_complete": True,
                    "end_call": False
                }))

    except WebSocketDisconnect:
        print(f"[CLOSE] call_id={call_id}")