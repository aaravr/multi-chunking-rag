import os
from typing import Any, Dict, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from core.contracts import DIResult


class DIClient:
    def __init__(self) -> None:
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
        api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        if not endpoint or not api_key:
            raise RuntimeError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY are required."
            )
        self._client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )

    def analyze_page_bytes(
        self,
        pdf_bytes: bytes,
        model_id: str = "prebuilt-layout",
        content_type: Optional[str] = "application/pdf",
    ) -> DIResult:
        poller = self._client.begin_analyze_document(
            model_id=model_id,
            body=pdf_bytes,
            content_type=content_type,
        )
        result = poller.result()
        return DIResult(result=_to_dict(result))

    def analyze_page_image_bytes(
        self, image_bytes: bytes, model_id: str = "prebuilt-layout"
    ) -> DIResult:
        return self.analyze_page_bytes(
            pdf_bytes=image_bytes,
            model_id=model_id,
            content_type="image/png",
        )


def _to_dict(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "as_dict"):
        return result.as_dict()
    if hasattr(result, "serialize"):
        return result.serialize()
    return dict(result)
