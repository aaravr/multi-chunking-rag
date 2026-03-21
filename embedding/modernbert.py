from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from core.config import settings


@dataclass
class TokenizedChunk:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offsets: List[Tuple[int, int]]


class ModernBERTEmbedder:
    def __init__(self, max_length: int = 8192) -> None:
        self.device = torch.device("cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.embedding_model, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            settings.embedding_model, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, text: str) -> TokenizedChunk:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
        offsets = [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"][0].tolist()
        ]
        return TokenizedChunk(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            offsets=offsets,
        )

    def tokenize_full(self, text: str) -> List[Tuple[int, int]]:
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=False,
        )
        return [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
        ]

    def encode(self, tokenized: TokenizedChunk) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
            )
        return output.last_hidden_state.squeeze(0)

    def embed_text(self, text: str) -> List[float]:
        tokenized = self.tokenize(text)
        embeddings = self.encode(tokenized)
        mask = tokenized.attention_mask.squeeze(0).unsqueeze(-1)
        masked = embeddings * mask
        pooled = masked.sum(dim=0) / mask.sum()
        return pooled.cpu().numpy().astype("float32").tolist()

    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single padded forward pass for throughput.

        Tokenizes all texts, pads to the longest sequence, runs one forward
        pass, then mean-pools each sequence individually.
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [self.embed_text(texts[0])]

        import numpy as np

        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
        hidden = output.last_hidden_state  # (batch, seq, dim)
        mask = encodings["attention_mask"].unsqueeze(-1)  # (batch, seq, 1)
        masked = hidden * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1)  # (batch, dim)
        return pooled.cpu().numpy().astype(np.float32).tolist()
