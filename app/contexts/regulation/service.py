from app.domain.models import Chunk


def filter_regulation_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.doc_type == "regulation"]
