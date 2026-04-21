import json
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

# --- CONFIGURACIÓN ---
OLLAMA_URL = "https://ia-tools-ollama.rzd02y.easypanel.host/"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_EMBED_MODEL = "llama3.2"  # el mismo modelo que usaste para indexar

QDRANT_URL = "https://ia-tools-qdrant.rzd02y.easypanel.host/"  # cambia por tu URL
QDRANT_API_KEY = "TU_API_KEY"                    # cambia por tu key
QDRANT_COLLECTION = "avidocumentation"                   # cambia por tu colección

SYSTEM_PROMPT = """Eres Gabriel, el asistente de CDV Consulting. 
Atiendes llamadas en español de forma profesional y concisa.
Responde siempre en español. Sé breve, máximo 2-3 frases, estás en una llamada telefónica.
Usa SOLO la información del contexto proporcionado para responder.
Si no sabes la respuesta, di que lo consultarás y que te dejen un contacto."""

OPENING_MESSAGE = "Hola, gracias por llamar a CDV Consulting. Soy Gabriel, ¿en qué puedo ayudarte?"


# --- FUNCIONES RAG ---

async def get_embedding(text: str) -> list:
    """Genera embedding con Ollama"""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
        )
    return response.json()["embedding"]


async def search_qdrant(query: str, top_k: int = 3) -> str:
    """Busca en Qdrant y devuelve el contexto relevante"""
    try:
        # Generar embedding de la pregunta
        embedding = await get_embedding(query)

        # Buscar en Qdrant
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

        results = response.json().get("result", [])

        if not results:
            return ""

        # Extraer texto de los resultados
        contexto = "\n\n".join([
            r["payload"].get("text", r["payload"].get("content", ""))
            for r in results
            if r.get("payload")
        ])

        print(f"[RAG] {len(results)} fragmentos encontrados")
        return contexto

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return ""


# --- WEBSOCKET HANDLER ---

@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"[OPEN] call_id={call_id}")

    # Gabriel habla primero
    await websocket.send_text(json.dumps({
        "response_type": "response",
        "response_id": 0,
        "content": OPENING_MESSAGE,
        "content_complete": True,
        "end_call": False
    }))
    print(f"[OUT_INIT] {OPENING_MESSAGE}")

    try:
        async for data in websocket.iter_text():
            request = json.loads(data)
            interaction_type = request.get("interaction_type")

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

                # Extraer última pregunta del usuario
                ultima_pregunta = ""
                for turn in reversed(transcript):
                    if turn["role"] == "user" and turn.get("content", "").strip():
                        ultima_pregunta = turn["content"].strip()
                        break

                print(f"[USER] {ultima_pregunta}")

                # Buscar contexto en Qdrant
                contexto = ""
                if ultima_pregunta:
                    contexto = await search_qdrant(ultima_pregunta)

                # Construir mensajes para Ollama
                system_con_contexto = SYSTEM_PROMPT
                if contexto:
                    system_con_contexto += f"\n\nCONTEXTO RELEVANTE:\n{contexto}"

                messages = [{"role": "system", "content": system_con_contexto}]

                for turn in transcript:
                    role = "assistant" if turn["role"] == "agent" else "user"
                    content = turn.get("content", "").strip()
                    if content:
                        messages.append({"role": role, "content": content})

                print(f"[OLLAMA] enviando {len(messages)} mensajes con contexto RAG")

                try:
                    async with httpx.AsyncClient(timeout=20) as client:
                        response = await client.post(
                            f"{OLLAMA_URL}/api/chat",
                            json={
                                "model": OLLAMA_MODEL,
                                "messages": messages,
                                "stream": False,
                                "options": {
                                    "num_predict": 100,
                                    "temperature": 0.5
                                }
                            }
                        )
                    reply = response.json()["message"]["content"]
                    print(f"[OLLAMA] respuesta: {reply}")

                except Exception as e:
                    print(f"[ERROR] Ollama: {e}")
                    reply = "Disculpa, un momento por favor."

                await websocket.send_text(json.dumps({
                    "response_type": "response",
                    "response_id": request.get("response_id"),
                    "content": reply,
                    "content_complete": True,
                    "end_call": False
                }))

    except WebSocketDisconnect:
        print(f"[CLOSE] call_id={call_id}")