import os
import json
import httpx

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("SERVER STARTED OK")

# =========================================================
# CONFIG
# =========================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "https://ia-tools-ollama.rzd02y.easypanel.host/"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "smollm2-realtime:latest"
)

OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://ia-tools-qdrant.rzd02y.easypanel.host"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "avidocumentation2"
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY", 
    "brMO3lyleeLXjUrr4UiTjxe8K9dlHEVk"
)


OPENING_MESSAGE = (
    "Hola, gracias por llamar a CDV Consulting. "
    "Soy Gabriel, ¿en qué puedo ayudarte?"
)

SYSTEM_PROMPT = """
Eres Gabriel, asistente de CDV Consulting.

Responde SIEMPRE en español.

IMPORTANTE:
- respuestas cortas
- máximo 2 frases
- tono natural
- no expliques demasiado
- evita respuestas largas

Usa el contexto si existe.
Si no sabes algo, dilo claramente.
"""


# =========================================================
# HEALTHCHECK
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "retell-websocket"
    }


@app.get("/health")
async def health():
    return {"ok": True}


# =========================================================
# EMBEDDINGS
# =========================================================

async def get_embedding(text: str) -> list:
    try:

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": OLLAMA_EMBED_MODEL,
                    "prompt": text
                }
            )

            response.raise_for_status()

        data = response.json()

        print("[EMBED RAW]", data)

        if "embedding" in data:
            return data["embedding"]

        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]

        print("[EMBED ERROR] formato desconocido")

        return []

    except Exception as e:

        print(f"[RAG ERROR - EMBEDDING] {e}")

        return []


# =========================================================
# QDRANT SEARCH
# =========================================================

async def search_qdrant(query: str, top_k: int = 3) -> str:

    try:

        embedding = await get_embedding(query)

        if not embedding:
            print("[RAG] embedding vacío")
            return ""

        headers = {
            "Content-Type": "application/json"
        }

        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                headers=headers,
                json={
                    "vector": embedding,
                    "limit": top_k,
                    "with_payload": True
                }
            )

            response.raise_for_status()

        data = response.json()

        print("[QDRANT RAW]", data)

        results = data.get("result", [])

        if not results:
            print("[RAG] sin resultados")
            return ""

        contexto = "\n\n".join([
            r.get("payload", {}).get("text")
            or r.get("payload", {}).get("content", "")
            for r in results
        ])

        print(f"[RAG] encontrados {len(results)} fragmentos")

        return contexto

    except Exception as e:

        print(f"[RAG ERROR - QDRANT] {e}")

        return ""


# =========================================================
# OLLAMA CHAT
# =========================================================

async def call_ollama(messages: list) -> str:

    try:

        async with httpx.AsyncClient(timeout=60) as client:

            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 60
                    }
                }
            )

            response.raise_for_status()

        data = response.json()

        print("[OLLAMA RAW]", data)

        if "message" in data:
            return data["message"]["content"]

        if "error" in data:
            return f"Error del modelo: {data['error']}"

        return (
            "Disculpa, ha habido un problema "
            "generando la respuesta."
        )

    except Exception as e:

        print(f"[OLLAMA ERROR] {e}")

        return (
            "Disculpa, ahora mismo tengo un problema técnico."
        )


# =========================================================
# WEBSOCKET
# =========================================================

# =========================================================
# WEBSOCKET
# =========================================================

@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(
    websocket: WebSocket,
    call_id: str
):

    await websocket.accept()

    print(f"[OPEN] call_id={call_id}")

    # =====================================================
    # MENSAJE INICIAL
    # =====================================================

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

            # =================================================
            # PING / PONG
            # =================================================

            if interaction_type == "ping_pong":

                await websocket.send_text(json.dumps({
                    "response_type": "ping_pong",
                    "timestamp": request.get("timestamp")
                }))

                continue

            # =================================================
            # UPDATE ONLY
            # =================================================

            if interaction_type == "update_only":
                continue

            # =================================================
            # RESPONSE REQUIRED
            # =================================================

            if interaction_type in (
                "response_required",
                "reminder_required"
            ):

                transcript = request.get("transcript", [])

                ultima_pregunta = ""

                for turn in reversed(transcript):

                    if (
                        turn["role"] == "user"
                        and turn.get("content", "").strip()
                    ):
                        ultima_pregunta = (
                            turn["content"].strip()
                        )
                        break

                print(f"[USER] {ultima_pregunta}")

                # =============================================
                # RAG
                # =============================================

                contexto = ""

                if ultima_pregunta:
                    contexto = await search_qdrant(
                        ultima_pregunta
                    )

                # =============================================
                # SYSTEM MESSAGE
                # =============================================

                system_msg = SYSTEM_PROMPT

                if contexto:
                    system_msg += (
                        f"\n\nCONTEXTO:\n{contexto}"
                    )

                messages = [
                    {
                        "role": "system",
                        "content": system_msg
                    }
                ]

                for turn in transcript:

                    role = (
                        "assistant"
                        if turn["role"] == "agent"
                        else "user"
                    )

                    content = (
                        turn.get("content", "").strip()
                    )

                    if content:
                        messages.append({
                            "role": role,
                            "content": content
                        })

                print(
                    f"[OLLAMA] enviando "
                    f"{len(messages)} mensajes"
                )

                # =============================================
                # CALL MODEL
                # =============================================

                reply = await call_ollama(messages)

                print(f"[REPLY] {reply}")

                # =============================================
                # SEND RESPONSE
                # =============================================

                await websocket.send_text(json.dumps({
                    "response_type": "response",
                    "response_id": request.get(
                        "response_id"
                    ),
                    "content": reply,
                    "content_complete": True,
                    "end_call": False
                }))

    except WebSocketDisconnect:

        print(f"[CLOSE] call_id={call_id}")

    except Exception as e:

        print(f"[WEBSOCKET ERROR] {e}")

        try:
            await websocket.close()
        except Exception:
            pass