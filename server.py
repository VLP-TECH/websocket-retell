import os
import json
import time
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
    "http://46.4.17.165:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:14b-instruct-q4_K_M"
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
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": OLLAMA_EMBED_MODEL,
                    "prompt": text
                }
            )
            response.raise_for_status()

        data = response.json()
        t1 = time.time()
        print(f"[TIMING] Embedding: {t1-t0:.2f}s")

        if "embedding" in data:
            return data["embedding"]

        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]

        print("[EMBED ERROR] formato desconocido")
        return []

    except Exception as e:
        t1 = time.time()
        print(f"[TIMING] Embedding FAILED: {t1-t0:.2f}s | Error: {e}")
        return []


# =========================================================
# QDRANT SEARCH
# =========================================================

async def search_qdrant(query: str, top_k: int = 3) -> str:
    t0 = time.time()
    try:
        embedding = await get_embedding(query)
        t1 = time.time()

        if not embedding:
            print("[RAG] embedding vacío")
            return ""

        headers = {
            "Content-Type": "application/json"
        }

        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY

        async with httpx.AsyncClient(timeout=5) as client:
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
        t2 = time.time()
        print(f"[TIMING] Qdrant search: {t2-t1:.2f}s | RAG total: {t2-t0:.2f}s")

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
        t1 = time.time()
        print(f"[TIMING] Qdrant FAILED: {t1-t0:.2f}s | Error: {e}")
        return ""


# =========================================================
# OLLAMA CHAT
# =========================================================

async def call_ollama(messages: list) -> str:
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "num_ctx": 2048,
                        "num_predict": 60,
                        "temperature": 0.1,
                        "top_k": 10,
                        "top_p": 0.7,
                        "repeat_penalty": 1.1
                    }
                }
            )
            response.raise_for_status()

        data = response.json()
        t1 = time.time()
        print(f"[TIMING] Ollama inference: {t1-t0:.2f}s")

        if "message" in data:
            return data["message"]["content"]

        if "error" in data:
            return f"Error del modelo: {data['error']}"

        return (
            "Disculpa, ha habido un problema "
            "generando la respuesta."
        )

    except Exception as e:
        t1 = time.time()
        print(f"[TIMING] Ollama FAILED: {t1-t0:.2f}s | Error: {e}")
        return (
            "Disculpa, ahora mismo tengo un problema técnico."
        )


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
                t_total_start = time.time()

                transcript = request.get("transcript", [])
                ultima_pregunta = ""

                for turn in reversed(transcript):
                    if (
                        turn["role"] == "user"
                        and turn.get("content", "").strip()
                    ):
                        ultima_pregunta = turn["content"].strip()
                        break

                print(f"[USER] {ultima_pregunta}")

                # =============================================
                # RAG
                # =============================================

                t_rag_start = time.time()
                contexto = ""
                if ultima_pregunta:
                    contexto = await search_qdrant(ultima_pregunta)
                t_rag_end = time.time()
                print(f"[TIMING] RAG completo: {t_rag_end - t_rag_start:.2f}s")

                # =============================================
                # SYSTEM MESSAGE
                # =============================================

                system_msg = SYSTEM_PROMPT
                if contexto:
                    system_msg += f"\n\nCONTEXTO:\n{contexto}"

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
                    content = turn.get("content", "").strip()
                    if content:
                        messages.append({
                            "role": role,
                            "content": content
                        })

                print(f"[OLLAMA] enviando {len(messages)} mensajes")

                # =============================================
                # CALL MODEL
                # =============================================

                t_model_start = time.time()
                reply = await call_ollama(messages)
                t_model_end = time.time()
                print(f"[TIMING] Modelo: {t_model_end - t_model_start:.2f}s")

                print(f"[REPLY] {reply}")

                t_total_end = time.time()
                print(f"[TIMING] *** TOTAL respuesta: {t_total_end - t_total_start:.2f}s ***")

                # =============================================
                # SEND RESPONSE
                # =============================================

                await websocket.send_text(json.dumps({
                    "response_type": "response",
                    "response_id": request.get("response_id"),
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