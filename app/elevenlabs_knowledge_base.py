"""ElevenLabs Knowledge Base management for document upload and retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to store part number mappings
MAPPINGS_FILE = Path(__file__).parent.parent / "kb_mappings.json"


class ElevenLabsKBError(Exception):
    """Custom exception for ElevenLabs Knowledge Base errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _load_mappings() -> Dict[str, str]:
    """Load part number mappings from JSON file."""
    if not MAPPINGS_FILE.exists():
        return {}
    try:
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load mappings: {e}")
        return {}


def _save_mappings(mappings: Dict[str, str]) -> None:
    """Save part number mappings to JSON file."""
    try:
        with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save mappings: {e}")
        raise ElevenLabsKBError(f"Failed to save mappings: {e}")


def store_part_number_mapping(document_id: str, part_number: str) -> None:
    """Store mapping between document_id and part_number."""
    mappings = _load_mappings()
    mappings[document_id] = part_number
    _save_mappings(mappings)
    logger.info(f"Stored mapping: {document_id} -> {part_number}")


def get_part_number(document_id: str) -> Optional[str]:
    """Get part number for a document_id."""
    mappings = _load_mappings()
    return mappings.get(document_id)


def extract_part_number_from_name(name: str) -> Optional[str]:
    """Extract part number from document name (e.g., 'Part_1234_Manual.pdf')."""
    import re
    # Try to extract part number from name patterns
    patterns = [
        r"Part[_\s-]?(\w+)",  # Part_1234, Part-1234, Part 1234
        r"(\w+)[_\s-]?Part",  # 1234_Part
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def create_knowledge_base(
    api_key: str, name: str = "Technical Support Knowledge Base"
) -> Dict[str, Any]:
    """
    Create a new Knowledge Base in ElevenLabs.
    
    Args:
        api_key: ElevenLabs API key
        name: Name for the Knowledge Base
        
    Returns:
        Dictionary with knowledge_base_id and other info
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Try different API paths for Knowledge Base creation
        # Method 1: Try conversational_ai.knowledge_base
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                kb = client.conversational_ai.knowledge_base.create(name=name)
                logger.info(f"Created Knowledge Base: {kb.id} - {name}")
                return {
                    "id": kb.id if hasattr(kb, 'id') else str(kb),
                    "name": getattr(kb, 'name', name),
                    "created_at": getattr(kb, "created_at", None),
                }
        except AttributeError:
            pass
        
        # Method 2: Try direct knowledge_base (if available in newer SDK)
        try:
            if hasattr(client, 'knowledge_base'):
                kb = client.knowledge_base.create(name=name)
                logger.info(f"Created Knowledge Base: {kb.id} - {name}")
                return {
                    "id": kb.id if hasattr(kb, 'id') else str(kb),
                    "name": getattr(kb, 'name', name),
                    "created_at": getattr(kb, "created_at", None),
                }
        except AttributeError:
            pass
        
        # If both methods fail, provide instructions
        raise ElevenLabsKBError(
            "Knowledge Base creation via API is not available in this SDK version. "
            "Please create the Knowledge Base manually from ElevenLabs Dashboard: "
            "https://elevenlabs.io/app/knowledge-base\n"
            "Then add the Knowledge Base ID to your .env file as ELEVENLABS_KNOWLEDGE_BASE_ID"
        )
        
    except ElevenLabsKBError:
        raise
    except Exception as e:
        logger.error(f"Failed to create Knowledge Base: {e}")
        raise ElevenLabsKBError(
            f"Failed to create Knowledge Base: {str(e)}\n"
            "Note: You may need to create the Knowledge Base manually from the Dashboard: "
            "https://elevenlabs.io/app/knowledge-base"
        ) from e


def upload_document_with_part_number(
    file_bytes: bytes,
    file_name: str,
    part_number: str,
    api_key: str,
    knowledge_base_id: str,
    custom_name: Optional[str] = None,
    mime_type: Optional[str] = None,
    parent_folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload document to ElevenLabs Knowledge Base with part number in name.
    
    Args:
        file_bytes: File content as bytes
        file_name: Original file name
        part_number: Part number to include in document name
        api_key: ElevenLabs API key
        knowledge_base_id: Knowledge Base ID
        custom_name: Optional custom name (if not provided, will be generated)
        mime_type: Optional MIME type (if not provided, will be guessed from file_name)
        parent_folder_id: Optional folder ID to upload document to specific folder
        
    Returns:
        Dictionary with document_id, name, and part_number
    """
    try:
        from elevenlabs.client import ElevenLabs
        from io import BytesIO
        import mimetypes
        import tempfile
        import os
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Extract file extension
        file_path = Path(file_name)
        extension = file_path.suffix
        
        # Generate name with part number
        if custom_name:
            doc_name = f"Part_{part_number}_{custom_name}"
        else:
            # Extract base name without extension
            base_name = file_path.stem
            doc_name = f"Part_{part_number}_{base_name}{extension}"
        
        # Guess MIME type if not provided
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_name)
            if not mime_type:
                # Fallback based on extension
                extension_lower = extension.lower() if extension else ''
                mime_map = {
                    '.pdf': 'application/pdf',
                    '.txt': 'text/plain',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.md': 'text/markdown',
                    '.html': 'text/html',
                    '.epub': 'application/epub+zip',
                }
                mime_type = mime_map.get(extension_lower, 'application/octet-stream')
        
        # Create a temporary file with proper extension so SDK can detect MIME type
        # ElevenLabs SDK may need the file to have proper extension for MIME type detection
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        # Upload document - use file path instead of BytesIO for better MIME type detection
        # Method 1: Try conversational_ai.knowledge_base
        # Note: create_from_file() doesn't take knowledge_base_id according to API docs
        # It takes: file, name, and optionally parent_folder_id
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                # Try with file path first (if SDK supports it)
                try:
                    with open(temp_file_path, 'rb') as f:
                        if parent_folder_id:
                            result = client.conversational_ai.knowledge_base.documents.create_from_file(
                                file=f,
                                name=doc_name,
                                parent_folder_id=parent_folder_id,
                            )
                        else:
                            result = client.conversational_ai.knowledge_base.documents.create_from_file(
                                file=f,
                                name=doc_name,
                            )
                except (TypeError, AttributeError):
                    # Fallback to BytesIO if file path doesn't work
                    file_obj = BytesIO(file_bytes)
                    if parent_folder_id:
                        result = client.conversational_ai.knowledge_base.documents.create_from_file(
                            file=file_obj,
                            name=doc_name,
                            parent_folder_id=parent_folder_id,
                        )
                    else:
                        result = client.conversational_ai.knowledge_base.documents.create_from_file(
                            file=file_obj,
                            name=doc_name,
                        )
            # Method 2: Try direct knowledge_base
            elif hasattr(client, 'knowledge_base'):
                try:
                    with open(temp_file_path, 'rb') as f:
                        if parent_folder_id:
                            result = client.knowledge_base.documents.create_from_file(
                                file=f,
                                name=doc_name,
                                parent_folder_id=parent_folder_id,
                            )
                        else:
                            result = client.knowledge_base.documents.create_from_file(
                                file=f,
                                name=doc_name,
                            )
                except (TypeError, AttributeError):
                    # Fallback to BytesIO if file path doesn't work
                    file_obj = BytesIO(file_bytes)
                    if parent_folder_id:
                        result = client.knowledge_base.documents.create_from_file(
                            file=file_obj,
                            name=doc_name,
                            parent_folder_id=parent_folder_id,
                        )
                    else:
                        result = client.knowledge_base.documents.create_from_file(
                            file=file_obj,
                            name=doc_name,
                        )
            else:
                raise AttributeError("Knowledge base API not found")
        except TypeError as e:
            # If there's a parameter error, provide helpful message
            if "knowledge_base_id" in str(e):
                raise ElevenLabsKBError(
                    f"API error: {str(e)}\n"
                    "Note: create_from_file() doesn't take knowledge_base_id parameter. "
                    "According to ElevenLabs API docs, it only takes: file, name, and optionally parent_folder_id."
                ) from e
            raise
        except AttributeError as e:
            raise ElevenLabsKBError(
                f"Knowledge Base API not available: {str(e)}\n"
                "Please ensure you're using the latest ElevenLabs SDK version."
            ) from e
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        
        # Store mapping
        document_id = result.id if hasattr(result, "id") else str(result)
        store_part_number_mapping(document_id, part_number)
        
        logger.info(f"Uploaded document: {doc_name} (ID: {document_id}, Part: {part_number}, MIME: {mime_type})")
        
        return {
            "document_id": document_id,
            "name": doc_name,
            "part_number": part_number,
            "original_name": file_name,
        }
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise ElevenLabsKBError(f"Failed to upload document: {str(e)}") from e


def upload_text_document_with_part_number(
    text: str,
    part_number: str,
    api_key: str,
    knowledge_base_id: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload text document to ElevenLabs Knowledge Base with part number.
    
    Args:
        text: Text content
        part_number: Part number
        api_key: ElevenLabs API key
        knowledge_base_id: Knowledge Base ID
        name: Optional document name
        
    Returns:
        Dictionary with document_id, name, and part_number
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Generate name with part number
        if not name:
            name = f"Part_{part_number}_Text_Document"
        else:
            name = f"Part_{part_number}_{name}"
        
        # Upload text document - try different API paths
        # Note: create_from_text() doesn't take knowledge_base_id according to API docs
        # It takes: text, name (optional), and optionally parent_folder_id
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                result = client.conversational_ai.knowledge_base.documents.create_from_text(
                    text=text,
                    name=name,
                )
            elif hasattr(client, 'knowledge_base'):
                result = client.knowledge_base.documents.create_from_text(
                    text=text,
                    name=name,
                )
            else:
                raise AttributeError("Knowledge base API not found")
        except TypeError as e:
            # If there's a parameter error, provide helpful message
            if "knowledge_base_id" in str(e):
                raise ElevenLabsKBError(
                    f"API error: {str(e)}\n"
                    "Note: create_from_text() doesn't take knowledge_base_id parameter. "
                    "According to ElevenLabs API docs, it only takes: text, name (optional), and optionally parent_folder_id."
                ) from e
            raise
        except AttributeError as e:
            raise ElevenLabsKBError(
                f"Knowledge Base API not available: {str(e)}\n"
                "Please ensure you're using the latest ElevenLabs SDK version."
            ) from e
        
        # Store mapping
        document_id = result.id if hasattr(result, "id") else str(result)
        store_part_number_mapping(document_id, part_number)
        
        logger.info(f"Uploaded text document: {name} (ID: {document_id}, Part: {part_number})")
        
        return {
            "document_id": document_id,
            "name": name,
            "part_number": part_number,
        }
    except Exception as e:
        logger.error(f"Failed to upload text document: {e}")
        raise ElevenLabsKBError(f"Failed to upload text document: {str(e)}") from e


def find_folder_by_name(
    api_key: str, folder_name: str
) -> Optional[str]:
    """
    Find a folder by name and return its ID.
    
    Args:
        api_key: ElevenLabs API key
        folder_name: Name of the folder to find
        
    Returns:
        Folder ID if found, None otherwise
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
            kb_client = client.conversational_ai.knowledge_base
            
            # List all items (including folders) - list() doesn't take knowledge_base_id
            kb_list_response = kb_client.list()
            all_items = getattr(kb_list_response, 'documents', []) or []
            
            # Search for folder by name
            for item in all_items:
                item_name = getattr(item, "name", "") or getattr(item, "file_name", "")
                item_type = getattr(item, 'type', None)
                
                # Check if it's a folder and name matches
                if item_name.lower() == folder_name.lower():
                    if item_type == 'folder' or (hasattr(item, 'children_count')):
                        return item.id if hasattr(item, "id") else (item.document_id if hasattr(item, "document_id") else None)
        
        return None
    except Exception as e:
        logger.warning(f"Failed to find folder: {e}")
        return None


def create_or_get_folder_by_part_number(
    api_key: str, part_number: str
) -> Optional[str]:
    """
    Create or get a folder by part number.
    Folder name will be: Part_{part_number}
    
    Args:
        api_key: ElevenLabs API key
        part_number: Part number to create folder for
        
    Returns:
        Folder ID if found or created, None if creation failed
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    folder_name = f"Part_{part_number}"
    
    # First, try to find existing folder
    folder_id = find_folder_by_name(api_key, folder_name)
    if folder_id:
        logger.info(f"Found existing folder: {folder_name} (ID: {folder_id})")
        return folder_id
    
    # If not found, try to create it
    try:
        client = ElevenLabs(api_key=api_key)
        
        # Try to create folder using different API paths
        if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
            kb_client = client.conversational_ai.knowledge_base
            
            # Try to create folder - check if there's a create_folder or similar method
            if hasattr(kb_client, 'folders') and hasattr(kb_client.folders, 'create'):
                folder = kb_client.folders.create(name=folder_name)
                folder_id = folder.id if hasattr(folder, "id") else (folder.document_id if hasattr(folder, "document_id") else None)
                logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
                return folder_id
            elif hasattr(kb_client, 'create_folder'):
                folder = kb_client.create_folder(name=folder_name)
                folder_id = folder.id if hasattr(folder, "id") else (folder.document_id if hasattr(folder, "document_id") else None)
                logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
                return folder_id
        elif hasattr(client, 'knowledge_base'):
            kb_client = client.knowledge_base
            
            if hasattr(kb_client, 'folders') and hasattr(kb_client.folders, 'create'):
                folder = kb_client.folders.create(name=folder_name)
                folder_id = folder.id if hasattr(folder, "id") else (folder.document_id if hasattr(folder, "document_id") else None)
                logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
                return folder_id
            elif hasattr(kb_client, 'create_folder'):
                folder = kb_client.create_folder(name=folder_name)
                folder_id = folder.id if hasattr(folder, "id") else (folder.document_id if hasattr(folder, "document_id") else None)
                logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
                return folder_id
        
        # If folder creation is not supported via API, log warning and return None
        # Documents can still be uploaded without a folder
        logger.warning(f"Folder creation not supported via API. Folder '{folder_name}' will not be created.")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to create folder '{folder_name}': {e}. Documents will be uploaded without folder.")
        return None


def list_documents(
    api_key: str, knowledge_base_id: str, parent_folder_id: Optional[str] = None, folder_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all documents in a Knowledge Base.
    
    Note: According to ElevenLabs API docs, list() doesn't take knowledge_base_id.
    It lists all accessible knowledge bases. Use parent_folder_id to filter by folder.
    
    Args:
        api_key: ElevenLabs API key
        knowledge_base_id: Knowledge Base ID (used for context, not passed to list())
        parent_folder_id: Optional folder ID to list documents from specific folder
        folder_name: Optional folder name to search for (will find folder and use its ID)
        
    Returns:
        List of documents with their info and part numbers
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # If folder_name is provided, find the folder first
        if folder_name and not parent_folder_id:
            folder_id = find_folder_by_name(api_key, folder_name)
            if folder_id:
                parent_folder_id = folder_id
                logger.info(f"Found folder '{folder_name}' with ID: {folder_id}")
            else:
                logger.warning(f"Folder '{folder_name}' not found, listing all documents")
        
        # List documents - list() doesn't take knowledge_base_id, it lists all accessible KBs
        # We need to filter or use parent_folder_id to get documents from specific folder
        all_documents = []
        
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                kb_client = client.conversational_ai.knowledge_base
                
                # Call list() without knowledge_base_id - it lists from all accessible KBs
                # Use parent_folder_id if provided to get documents from specific folder
                list_params = {}
                if parent_folder_id:
                    list_params['parent_folder_id'] = parent_folder_id
                
                # Get all documents (may need pagination with cursor)
                cursor = None
                while True:
                    if cursor:
                        list_params['cursor'] = cursor
                    
                    kb_list_response = kb_client.list(**list_params)
                    
                    # Extract documents from response
                    docs_batch = getattr(kb_list_response, 'documents', []) or []
                    all_documents.extend(docs_batch)
                    
                    # Check if there are more pages
                    has_more = getattr(kb_list_response, 'has_more', False)
                    cursor = getattr(kb_list_response, 'next_cursor', None)
                    
                    if not has_more or not cursor:
                        break
                
            elif hasattr(client, 'knowledge_base'):
                kb_client = client.knowledge_base
                list_params = {}
                if parent_folder_id:
                    list_params['parent_folder_id'] = parent_folder_id
                
                cursor = None
                while True:
                    if cursor:
                        list_params['cursor'] = cursor
                    
                    kb_list_response = kb_client.list(**list_params)
                    docs_batch = getattr(kb_list_response, 'documents', []) or []
                    all_documents.extend(docs_batch)
                    
                    has_more = getattr(kb_list_response, 'has_more', False)
                    cursor = getattr(kb_list_response, 'next_cursor', None)
                    
                    if not has_more or not cursor:
                        break
            else:
                raise AttributeError("Knowledge base API not found")
        except AttributeError as e:
            raise ElevenLabsKBError(
                f"Knowledge Base API not available: {str(e)}\n"
                "Please ensure you're using the latest ElevenLabs SDK version."
            ) from e
        
        # Load mappings to get part numbers
        mappings = _load_mappings()
        
        result = []
        for doc in all_documents:
            # Skip folders - only include actual documents
            doc_type = getattr(doc, 'type', None)
            if doc_type == 'folder' or hasattr(doc, 'children_count'):
                continue
            
            doc_id = doc.id if hasattr(doc, "id") else (doc.document_id if hasattr(doc, "document_id") else str(doc))
            doc_name = getattr(doc, "name", "") or getattr(doc, "file_name", "") or getattr(doc, "title", "")
            part_number = mappings.get(doc_id) or extract_part_number_from_name(doc_name)
            
            # Get size from metadata if available
            metadata = getattr(doc, 'metadata', None)
            size = None
            if metadata:
                size = getattr(metadata, 'size_bytes', None)
            if not size:
                size = getattr(doc, "size", None)
            
            result.append({
                "document_id": doc_id,
                "name": doc_name,
                "part_number": part_number,
                "size": size,
                "created_at": getattr(doc, "created_at", None),
                "type": doc_type,
            })
        
        return result
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise ElevenLabsKBError(f"Failed to list documents: {str(e)}") from e


def get_document_info(
    api_key: str, knowledge_base_id: str, document_id: str
) -> Dict[str, Any]:
    """
    Get information about a specific document.
    
    Args:
        api_key: ElevenLabs API key
        knowledge_base_id: Knowledge Base ID
        document_id: Document ID
        
    Returns:
        Document information with part number
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Get document - try different API paths
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                doc = client.conversational_ai.knowledge_base.documents.get(
                    knowledge_base_id=knowledge_base_id, document_id=document_id
                )
            elif hasattr(client, 'knowledge_base'):
                doc = client.knowledge_base.documents.get(
                    knowledge_base_id=knowledge_base_id, document_id=document_id
                )
            else:
                raise AttributeError("Knowledge base API not found")
        except AttributeError as e:
            raise ElevenLabsKBError(
                f"Knowledge Base API not available: {str(e)}\n"
                "Please ensure you're using the latest ElevenLabs SDK version."
            ) from e
        
        # Get part number from mappings
        mappings = _load_mappings()
        part_number = mappings.get(document_id) or extract_part_number_from_name(
            getattr(doc, "name", "")
        )
        
        return {
            "document_id": document_id,
            "name": getattr(doc, "name", ""),
            "part_number": part_number,
            "size": getattr(doc, "size", None),
            "created_at": getattr(doc, "created_at", None),
        }
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise ElevenLabsKBError(f"Failed to get document info: {str(e)}") from e


def delete_document(
    api_key: str, knowledge_base_id: str, document_id: str
) -> None:
    """
    Delete a document from Knowledge Base.
    
    Args:
        api_key: ElevenLabs API key
        knowledge_base_id: Knowledge Base ID
        document_id: Document ID to delete
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Delete document - try different API paths
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                client.conversational_ai.knowledge_base.documents.delete(
                    knowledge_base_id=knowledge_base_id, document_id=document_id
                )
            elif hasattr(client, 'knowledge_base'):
                client.knowledge_base.documents.delete(
                    knowledge_base_id=knowledge_base_id, document_id=document_id
                )
            else:
                raise AttributeError("Knowledge base API not found")
        except AttributeError as e:
            raise ElevenLabsKBError(
                f"Knowledge Base API not available: {str(e)}\n"
                "Please ensure you're using the latest ElevenLabs SDK version."
            ) from e
        
        # Remove from mappings
        mappings = _load_mappings()
        if document_id in mappings:
            del mappings[document_id]
            _save_mappings(mappings)
        
        logger.info(f"Deleted document: {document_id}")
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise ElevenLabsKBError(f"Failed to delete document: {str(e)}") from e


def assign_knowledge_base_to_agent(
    api_key: str, agent_id: str, knowledge_base_id: str
) -> Dict[str, Any]:
    """
    Assign Knowledge Base to an Agent.
    Tries multiple methods: SDK update, REST API, and direct HTTP calls.
    
    Args:
        api_key: ElevenLabs API key
        agent_id: Agent ID
        knowledge_base_id: Knowledge Base ID to assign
        
    Returns:
        Dictionary with assignment result
    """
    try:
        import httpx
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs" or "httpx". Install it with: pip install elevenlabs httpx'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Method 1: Try using SDK with different parameter names
        try:
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'agents'):
                agent = client.conversational_ai.agents.get(agent_id=agent_id)
                
                # Try update with different parameter variations
                update_params = {}
                
                # Try knowledge_base_id
                try:
                    updated = client.conversational_ai.agents.update(
                        agent_id=agent_id,
                        knowledge_base_id=knowledge_base_id,
                    )
                    logger.info(f"✅ Assigned Knowledge Base {knowledge_base_id} to Agent {agent_id} via SDK")
                    return {
                        "agent_id": agent_id,
                        "knowledge_base_id": knowledge_base_id,
                        "status": "success",
                        "method": "sdk_update",
                    }
                except TypeError:
                    pass
                
                # Try knowledge_bases (plural, as list)
                try:
                    updated = client.conversational_ai.agents.update(
                        agent_id=agent_id,
                        knowledge_bases=[knowledge_base_id],
                    )
                    logger.info(f"✅ Assigned Knowledge Base {knowledge_base_id} to Agent {agent_id} via SDK (knowledge_bases)")
                    return {
                        "agent_id": agent_id,
                        "knowledge_base_id": knowledge_base_id,
                        "status": "success",
                        "method": "sdk_update_knowledge_bases",
                    }
                except (TypeError, AttributeError):
                    pass
                
                # Try config parameter
                try:
                    updated = client.conversational_ai.agents.update(
                        agent_id=agent_id,
                        config={"knowledge_base_id": knowledge_base_id},
                    )
                    logger.info(f"✅ Assigned Knowledge Base {knowledge_base_id} to Agent {agent_id} via SDK (config)")
                    return {
                        "agent_id": agent_id,
                        "knowledge_base_id": knowledge_base_id,
                        "status": "success",
                        "method": "sdk_update_config",
                    }
                except (TypeError, AttributeError):
                    pass
        except Exception as e:
            logger.debug(f"SDK method failed: {e}")
        
        # Method 2: Try REST API directly
        try:
            base_url = "https://api.elevenlabs.io/v1"
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            }
            
            # Try PATCH /v1/conversational-ai/agents/{agent_id}
            with httpx.Client(timeout=30.0) as http_client:
                # Method 2a: Try PATCH with knowledge_base_id
                try:
                    response = http_client.patch(
                        f"{base_url}/conversational-ai/agents/{agent_id}",
                        headers=headers,
                        json={"knowledge_base_id": knowledge_base_id},
                    )
                    if response.status_code == 200:
                        logger.info(f"✅ Assigned Knowledge Base via REST API (PATCH)")
                        return {
                            "agent_id": agent_id,
                            "knowledge_base_id": knowledge_base_id,
                            "status": "success",
                            "method": "rest_api_patch",
                        }
                except Exception as e:
                    logger.debug(f"REST PATCH failed: {e}")
                
                # Method 2b: Try PUT
                try:
                    response = http_client.put(
                        f"{base_url}/conversational-ai/agents/{agent_id}",
                        headers=headers,
                        json={"knowledge_base_id": knowledge_base_id},
                    )
                    if response.status_code == 200:
                        logger.info(f"✅ Assigned Knowledge Base via REST API (PUT)")
                        return {
                            "agent_id": agent_id,
                            "knowledge_base_id": knowledge_base_id,
                            "status": "success",
                            "method": "rest_api_put",
                        }
                except Exception as e:
                    logger.debug(f"REST PUT failed: {e}")
                
                # Method 2c: Try POST to assign endpoint
                try:
                    response = http_client.post(
                        f"{base_url}/conversational-ai/agents/{agent_id}/knowledge-bases",
                        headers=headers,
                        json={"knowledge_base_id": knowledge_base_id},
                    )
                    if response.status_code in [200, 201]:
                        logger.info(f"✅ Assigned Knowledge Base via REST API (POST)")
                        return {
                            "agent_id": agent_id,
                            "knowledge_base_id": knowledge_base_id,
                            "status": "success",
                            "method": "rest_api_post",
                        }
                except Exception as e:
                    logger.debug(f"REST POST failed: {e}")
                
                # Method 2d: Try with knowledge_bases array
                try:
                    response = http_client.patch(
                        f"{base_url}/conversational-ai/agents/{agent_id}",
                        headers=headers,
                        json={"knowledge_bases": [knowledge_base_id]},
                    )
                    if response.status_code == 200:
                        logger.info(f"✅ Assigned Knowledge Base via REST API (knowledge_bases array)")
                        return {
                            "agent_id": agent_id,
                            "knowledge_base_id": knowledge_base_id,
                            "status": "success",
                            "method": "rest_api_patch_array",
                        }
                except Exception as e:
                    logger.debug(f"REST PATCH array failed: {e}")
        except Exception as e:
            logger.debug(f"REST API methods failed: {e}")
        
        # Method 3: Try direct agents endpoint (not conversational_ai)
        try:
            if hasattr(client, 'agents'):
                agent = client.agents.get(agent_id=agent_id)
                
                # Try update with knowledge_base_id
                try:
                    updated = client.agents.update(
                        agent_id=agent_id,
                        knowledge_base_id=knowledge_base_id,
                    )
                    logger.info(f"✅ Assigned Knowledge Base via direct agents.update()")
                    return {
                        "agent_id": agent_id,
                        "knowledge_base_id": knowledge_base_id,
                        "status": "success",
                        "method": "direct_agents_update",
                    }
                except (TypeError, AttributeError) as e:
                    logger.debug(f"Direct agents.update() failed: {e}")
        except Exception as e:
            logger.debug(f"Direct agents method failed: {e}")
        
        # If all methods fail, return manual assignment instructions
        logger.warning(
            f"All programmatic methods failed. Knowledge Base assignment must be done via Dashboard. "
            f"Agent: {agent_id}, KB: {knowledge_base_id}"
        )
        return {
            "agent_id": agent_id,
            "knowledge_base_id": knowledge_base_id,
            "status": "manual_required",
            "note": (
                "⚠️ ElevenLabs API doesn't support assigning Knowledge Base to Agent programmatically.\n\n"
                "📋 **Please follow these steps:**\n"
                "1. Go to ElevenLabs Dashboard: https://elevenlabs.io/app/agents\n"
                f"2. Open Agent: `{agent_id}`\n"
                "3. Go to 'Knowledge Base' or 'Settings' section\n"
                f"4. Select Knowledge Base: `{knowledge_base_id}`\n"
                "5. Save the changes\n\n"
                "✅ After assignment, the Agent will automatically use the Knowledge Base for RAG."
            ),
            "dashboard_url": f"https://elevenlabs.io/app/agents/{agent_id}",
        }
        
    except Exception as e:
        logger.error(f"Failed to assign Knowledge Base to Agent: {e}")
        raise ElevenLabsKBError(f"Failed to assign Knowledge Base to Agent: {str(e)}") from e


def sync_rag_index(
    api_key: str, knowledge_base_id: str, document_id: Optional[str] = None
) -> None:
    """
    Sync RAG index for Knowledge Base or specific document.
    
    Args:
        api_key: ElevenLabs API key
        knowledge_base_id: Knowledge Base ID
        document_id: Optional specific document ID (if None, syncs all)
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e

    try:
        client = ElevenLabs(api_key=api_key)
        
        # Sync RAG index - try different API paths
        try:
            kb_docs = None
            if hasattr(client, 'conversational_ai') and hasattr(client.conversational_ai, 'knowledge_base'):
                kb_docs = client.conversational_ai.knowledge_base.documents
            elif hasattr(client, 'knowledge_base'):
                kb_docs = client.knowledge_base.documents
            else:
                raise AttributeError("Knowledge base API not found")
            
            if document_id:
                # Sync specific document
                if hasattr(kb_docs, 'compute_rag_index'):
                    kb_docs.compute_rag_index(
                        knowledge_base_id=knowledge_base_id, document_id=document_id
                    )
                    logger.info(f"Synced RAG index for document: {document_id}")
                else:
                    logger.warning("compute_rag_index method not available - RAG sync may happen automatically")
            else:
                # Sync all documents (if API supports it)
                documents = list_documents(api_key, knowledge_base_id)
                for doc in documents:
                    try:
                        if hasattr(kb_docs, 'compute_rag_index'):
                            kb_docs.compute_rag_index(
                                knowledge_base_id=knowledge_base_id,
                                document_id=doc["document_id"],
                            )
                    except Exception as e:
                        logger.warning(f"Failed to sync document {doc['document_id']}: {e}")
                
                logger.info(f"Synced RAG index for {len(documents)} documents")
        except AttributeError as e:
            logger.warning(f"RAG sync API not available: {str(e)} - RAG may sync automatically")
    except Exception as e:
        logger.error(f"Failed to sync RAG index: {e}")
        raise ElevenLabsKBError(f"Failed to sync RAG index: {str(e)}") from e
