"""Google Form webhook integration for automatic document ingestion."""

import asyncio
import hashlib
import hmac
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from fastapi import BackgroundTasks, HTTPException, Request

from ..utils import get_logger, get_settings
from .pdf_to_markdown import convert_pdf_to_markdown

logger = get_logger(__name__)
settings = get_settings()


class GoogleFormWebhook:
    """
    Handle Google Form submissions for document ingestion.
    
    Setup instructions:
    1. Create a Google Form with file upload field
    2. Create a Google Apps Script trigger on form submit
    3. Configure the script to POST to this webhook endpoint
    
    Example Apps Script:
    ```javascript
    function onFormSubmit(e) {
        const responses = e.response.getItemResponses();
        const fileUpload = responses.find(r => 
            r.getItem().getType() === FormApp.ItemType.FILE_UPLOAD
        );
        
        if (fileUpload) {
            const fileIds = fileUpload.getResponse();
            const files = fileIds.map(id => {
                const file = DriveApp.getFileById(id);
                return {
                    name: file.getName(),
                    url: file.getDownloadUrl(),
                    mimeType: file.getMimeType()
                };
            });
            
            const payload = {
                timestamp: new Date().toISOString(),
                files: files,
                email: e.response.getRespondentEmail()
            };
            
            UrlFetchApp.fetch("https://your-api.com/api/v1/webhook/google-form", {
                method: "POST",
                contentType: "application/json",
                payload: JSON.stringify(payload),
                headers: {
                    "X-Webhook-Secret": "your-secret-key"
                }
            });
        }
    }
    ```
    """

    def __init__(
        self,
        webhook_secret: Optional[str] = None,
        download_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize webhook handler.
        
        Args:
            webhook_secret: Secret for webhook verification
            download_dir: Directory for downloaded files
            output_dir: Directory for processed output
        """
        self.webhook_secret = webhook_secret
        self.download_dir = download_dir or Path(tempfile.mkdtemp())
        self.output_dir = output_dir or settings.ingestion.markdown_dir
        
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def verify_signature(self, request: Request, body: bytes) -> bool:
        """
        Verify webhook signature.
        
        Args:
            request: FastAPI request object
            body: Request body bytes
            
        Returns:
            True if signature is valid
        """
        if not self.webhook_secret:
            logger.warning("No webhook secret configured - skipping verification")
            return True
            
        signature = request.headers.get("X-Webhook-Secret")
        if not signature:
            return False
            
        expected = hmac.new(
            self.webhook_secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)

    async def handle_submission(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> dict:
        """
        Handle incoming Google Form submission.
        
        Args:
            request: FastAPI request
            background_tasks: FastAPI background tasks
            
        Returns:
            Response dictionary
        """
        body = await request.body()
        
        # Verify signature
        if not self.verify_signature(request, body):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        files = data.get("files", [])
        if not files:
            return {"status": "ok", "message": "No files to process"}
        
        # Queue background processing
        submission_id = f"form_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        background_tasks.add_task(
            self._process_files,
            files=files,
            submission_id=submission_id,
            email=data.get("email"),
        )
        
        return {
            "status": "accepted",
            "submission_id": submission_id,
            "files_queued": len(files),
        }

    async def _process_files(
        self,
        files: list[dict],
        submission_id: str,
        email: Optional[str] = None,
    ) -> None:
        """
        Process uploaded files in background.
        
        Args:
            files: List of file info dicts
            submission_id: Unique submission identifier
            email: Submitter email for notifications
        """
        logger.info(f"Processing submission {submission_id} with {len(files)} files")
        
        results = []
        async with aiohttp.ClientSession() as session:
            for file_info in files:
                try:
                    result = await self._download_and_process(
                        session=session,
                        file_info=file_info,
                        submission_id=submission_id,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process file {file_info.get('name')}: {e}")
                    results.append({
                        "file": file_info.get("name"),
                        "success": False,
                        "error": str(e),
                    })
        
        # Log results
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(
            f"Submission {submission_id} complete: "
            f"{success_count}/{len(results)} files processed"
        )
        
        # TODO: Send notification email if configured

    async def _download_and_process(
        self,
        session: aiohttp.ClientSession,
        file_info: dict,
        submission_id: str,
    ) -> dict:
        """
        Download and process a single file.
        
        Args:
            session: aiohttp session
            file_info: File information dict
            submission_id: Submission identifier
            
        Returns:
            Processing result dict
        """
        file_name = file_info.get("name", "unknown")
        file_url = file_info.get("url")
        mime_type = file_info.get("mimeType", "")
        
        if not file_url:
            return {"file": file_name, "success": False, "error": "No URL provided"}
        
        # Only process PDFs
        if not (file_name.lower().endswith(".pdf") or "pdf" in mime_type.lower()):
            return {
                "file": file_name,
                "success": False,
                "error": "Not a PDF file",
            }
        
        # Download file
        download_path = self.download_dir / f"{submission_id}_{file_name}"
        
        try:
            async with session.get(file_url) as response:
                if response.status != 200:
                    return {
                        "file": file_name,
                        "success": False,
                        "error": f"Download failed: HTTP {response.status}",
                    }
                
                content = await response.read()
                download_path.write_bytes(content)
                
        except Exception as e:
            return {"file": file_name, "success": False, "error": f"Download error: {e}"}
        
        # Convert to markdown
        try:
            result = convert_pdf_to_markdown(
                pdf_path=download_path,
                output_dir=self.output_dir / submission_id,
            )
            
            # Clean up download
            download_path.unlink(missing_ok=True)
            
            return {
                "file": file_name,
                "success": result.get("success", False),
                "output": result.get("output_path"),
            }
            
        except Exception as e:
            return {"file": file_name, "success": False, "error": f"Conversion error: {e}"}


# Global webhook handler instance
_webhook_handler: Optional[GoogleFormWebhook] = None


def get_webhook_handler() -> GoogleFormWebhook:
    """Get or create webhook handler instance."""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = GoogleFormWebhook()
    return _webhook_handler


def configure_webhook(
    webhook_secret: Optional[str] = None,
    download_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> GoogleFormWebhook:
    """Configure and return webhook handler."""
    global _webhook_handler
    _webhook_handler = GoogleFormWebhook(
        webhook_secret=webhook_secret,
        download_dir=download_dir,
        output_dir=output_dir,
    )
    return _webhook_handler
