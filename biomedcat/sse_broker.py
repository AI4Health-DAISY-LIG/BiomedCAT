import asyncio
import json
from typing import AsyncGenerator
from biomedcat.events import event_emitter

class SSEBroker:
    """
    Le Broker fait le pont entre l'EventEmitter synchrone (utilisé par le pipeline)
    et un flux de données asynchrone (utilisé par FastAPI pour le SSE).
    """
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        # On s'abonne à l'event_emitter global pour intercepter les événements du pipeline
        event_emitter.subscribe(self._handle_event)

    def _handle_event(self, event_name: str, data: dict) -> None:
        """
        Callback synchrone appelé par le pipeline (souvent dans un thread séparé).
        Utilise call_soon_threadsafe pour injecter l'événement de manière sécurisée 
        dans la queue asyncio du thread principal.
        """
        try:
            # On récupère le loop en cours de l'application FastAPI
            loop = asyncio.get_running_loop()
            # On prépare le payload (nom de l'event et données)
            payload = (event_name, data)
            # Injection sécurisée dans la queue depuis un autre thread
            loop.call_soon_threadsafe(self.queue.put_nowait, payload)
        except RuntimeError:
            # Si aucun loop n'est en cours (ex: au démarrage de l'app), on ignore l'événement
            pass

    async def event_generator(self) -> AsyncGenerator[str, None]:
        """
        Générateur asynchrone pour la StreamingResponse de FastAPI.
        Formate les données selon le protocole Server-Sent Events (SSE).
        """
        while True:
            # On attend le prochain événement dans la queue
            event_name, data = await self.queue.get()
            
            # Formatage strict pour le protocole SSE :
            # event: <nom>\n
            # data: <json>\n\n
            yield f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

# Instance unique (Singleton) à importer dans votre route FastAPI
sse_broker = SSEBroker()
