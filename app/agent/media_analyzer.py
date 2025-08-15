from pydantic import BaseModel, Field
from app.config import get_gemini_model
from pydantic_ai import Agent, BinaryContent, StructuredDict
from typing import List, Optional
import os
import mimetypes
import aiofiles
import asyncio
import json
import pandas as pd
from pathlib import Path
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ----------------------------
#   Pydantic Output Schema
# ----------------------------


class MediaMetadata(BaseModel):
    """Structured description of extracted metadata & insights from media."""
    filename: str = Field(...,
                          description="Original filename of the media file")
    media_type: str = Field(...,
                            description="Detected media type: image, audio, video, pdf")
    description: str = Field(...,
                             description="High-level semantic description")
    detected_objects: List[str] = Field(
        default_factory=list, description="List of distinct entities, objects, or features present")
    textual_content: Optional[str] = Field(
        None, description="Detected text (OCR or embedded captions)")
    statistical_info: Optional[str] = Field(
        None, description="Any quantifiable data (counts, measurements, dimensions)")
    possible_analytical_uses: List[str] = Field(
        default_factory=list, description="Ways this media could be used in a data analysis pipeline")
    has_tables: Optional[bool] = Field(
        False, description="Whether the content contains tables that should be extracted")
    extract_all_tables: Optional[bool] = Field(
        False, description="Whether all tables should be extracted from the document")


class TableExtractionTool(BaseModel):
    """Tool to extract tables from PDF documents."""
    should_extract_tables: bool = Field(...,
                                        description="Whether to extract all tables from the PDF")
    reason: str = Field(...,
                        description="Reason why tables should or should not be extracted")


# Optional: get JSON schema from Pydantic
mediametadata_schema = MediaMetadata.model_json_schema()


# Create Agent function to ensure it's created in the correct event loop
def create_agent():
    """Create agent with proper event loop context"""
    return Agent(
        output_type=StructuredDict(mediametadata_schema),
        model=get_gemini_model()
    )


# ----------------------------
#   PDF Processing Functions
# ----------------------------

async def extract_pdf_first_page_text(pdf_path: str) -> Optional[str]:
    """Extract text from the first page of a PDF."""
    if not fitz:
        print("PyMuPDF not available, skipping PDF text extraction")
        return None

    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None

        first_page = doc[0]
        text = first_page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return None


async def extract_all_pdf_tables(pdf_path: str, temp_dir: str) -> Optional[str]:
    """Extract all tables from PDF and save as CSV."""
    if not fitz:
        print("PyMuPDF not available, skipping PDF table extraction")
        return None

    try:
        doc = fitz.open(pdf_path)
        all_tables = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Try to find tables using PyMuPDF's table detection
            try:
                tables = page.find_tables()
                for table in tables:
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:  # At least header + 1 row
                        # Convert to DataFrame
                        df = pd.DataFrame(
                            table_data[1:], columns=table_data[0])
                        df['page_number'] = page_num + 1
                        all_tables.append(df)
            except Exception as table_error:
                print(
                    f"Error extracting tables from page {page_num + 1}: {table_error}")
                continue

        doc.close()

        if all_tables:
            # Combine all tables
            combined_df = pd.concat(all_tables, ignore_index=True)

            # Create temp directory if it doesn't exist
            temp_path = Path(temp_dir)
            temp_path.mkdir(exist_ok=True)

            # Save as CSV
            filename = Path(pdf_path).stem
            csv_path = temp_path / f"{filename}_tables.csv"
            combined_df.to_csv(csv_path, index=False)

            print(
                f"Extracted {len(all_tables)} tables from PDF and saved to {csv_path}")
            return str(csv_path)
        else:
            print("No tables found in PDF")
            return None

    except Exception as e:
        print(f"Error extracting tables from PDF {pdf_path}: {e}")
        return None


async def create_table_extraction_agent():
    """Create agent for table extraction decision."""
    table_extraction_schema = TableExtractionTool.model_json_schema()
    return Agent(
        output_type=StructuredDict(table_extraction_schema),
        model=get_gemini_model()
    )


async def should_extract_tables(text_content: str, filename: str) -> bool:
    """Determine if tables should be extracted from the PDF based on text content."""
    agent = await create_table_extraction_agent()

    prompt = f"""
    Analyze the following text content from the first page of PDF file '{filename}' and determine if it contains references to tables that should be extracted.

    Look for indicators such as:
    - Mentions of "table", "tables"
    - References to "next page", "following page", "see page X"
    - Data structures, numerical data layouts
    - Column headers or tabular formatting
    - Financial reports, statistical data, charts

    Text content:
    {text_content[:2000]}  # Limit to first 2000 chars

    Determine if all tables should be extracted from this PDF.
    """

    try:
        result = await agent.run(prompt)
        return result.output.get('should_extract_tables', False)
    except Exception as e:
        print(f"Error in table extraction decision: {e}")
        return False


# ----------------------------
#   Helper: Read File as Bytes
# ----------------------------


async def _read_file_as_bytes(path: str) -> bytes:
    try:
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    except Exception as e:
        raise IOError(f"Failed to read file {path}: {str(e)}")

# ----------------------------
#   Helper: Construct Analysis Prompt
# ----------------------------


async def _construct_prompt(fname: str, media_type: str) -> str:
    prompt = f"""
You are a highly skilled data analyst specializing in extracting structured, data-ready insights from {media_type} content.
Analyze the file '{fname}' and produce valid JSON that matches the required schema.

Please extract the following information:
- filename: The original filename
- media_type: The type of media (image, audio, or video)
- description: A high-level semantic description of the content
- detected_objects: List of distinct entities, objects, or features present
- textual_content: Any detected text (OCR or embedded captions) if present
- statistical_info: Any quantifiable data like counts, measurements, dimensions
- possible_analytical_uses: Ways this media could be used in data analysis

Return the analysis as a properly structured JSON object.
"""
    return prompt


# ----------------------------
#   Main Analyzer
# ----------------------------


async def analyze_media(file_path: str, temp_dir: str = "tmp") -> Optional[MediaMetadata]:
    print(f"*************Analyzing media file: {file_path}")
    mime_type, _ = mimetypes.guess_type(str(file_path))
    print(f"Detected mime_type for {file_path}: {mime_type}")
    if not mime_type:
        mime_type = "application/octet-stream"

    # Handle different file types
    if mime_type.startswith("image"):
        print(f"Processing image file: {file_path}")
        media_type = "image"
        return await _process_media_file(file_path, media_type, mime_type)
    elif mime_type.startswith("audio"):
        print(f"Processing audio file: {file_path}")
        media_type = "audio"
        return await _process_media_file(file_path, media_type, mime_type)
    elif mime_type.startswith("video"):
        print(f"Processing video file: {file_path}")
        media_type = "video"
        return await _process_media_file(file_path, media_type, mime_type)
    elif mime_type == "application/pdf" or str(file_path).lower().endswith('.pdf'):
        print(f"Processing PDF file: {file_path}")
        return await _process_pdf_file(file_path, temp_dir)
    else:
        print(f"Skipping unsupported file type for {file_path}: {mime_type}")
        return None


async def _process_media_file(file_path: str, media_type: str, mime_type: str) -> Optional[MediaMetadata]:
    """Process image, audio, or video files."""
    try:
        # Create agent within the function to ensure proper event loop context
        agent = create_agent()

        file_bytes = await _read_file_as_bytes(str(file_path))
        print(f"Read {len(file_bytes)} bytes from {file_path}")
        media_content = BinaryContent(data=file_bytes, media_type=mime_type)

        prompt = await _construct_prompt(str(file_path), media_type)
        print(f"Generated prompt for {file_path}: {prompt[:200]}...")

        result = await agent.run([
            prompt, media_content
        ])

        print(f"Raw agent output for {file_path}: {result.output}")
        print(f"Output type: {type(result.output)}")

        if not isinstance(result.output, dict):
            print(
                f"ERROR: Agent output is not a dict for {file_path}. Got {type(result.output)}")
            return None

        try:
            metadata = MediaMetadata(**result.output)
            print(f"Successfully serialized MediaMetadata for {file_path}")
            return metadata
        except Exception as serialize_error:
            print(f"SERIALIZATION ERROR for {file_path}: {serialize_error}")
            print(f"Raw output was: {result.output}")
            return None

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def _process_pdf_file(file_path: str, temp_dir: str) -> Optional[MediaMetadata]:
    """Process PDF files with text and table extraction."""
    try:
        filename = Path(file_path).name

        # Extract text from first page
        first_page_text = await extract_pdf_first_page_text(file_path)
        if not first_page_text:
            print(f"No text extracted from first page of {file_path}")
            return None

        print(
            f"Extracted text from first page of {file_path}: {first_page_text[:500]}...")

        # Check if we should extract tables
        should_extract = await should_extract_tables(first_page_text, filename)
        print(f"Should extract tables from {file_path}: {should_extract}")

        csv_path = None
        if should_extract:
            csv_path = await extract_all_pdf_tables(file_path, temp_dir)
            if csv_path:
                print(f"Tables extracted and saved to: {csv_path}")
                # Re-analyze the extracted CSV file
                try:
                    from app.agent.file_analysis_handler import analyze_files_list
                    analysis_results = await analyze_files_list([csv_path])
                    print(f"Re-analyzed extracted tables: {analysis_results}")
                except Exception as analysis_error:
                    print(
                        f"Error re-analyzing extracted tables: {analysis_error}")

        # Create agent for PDF analysis
        agent = create_agent()

        prompt = f"""
        You are analyzing a PDF document. Extract structured metadata from this PDF.
        
        Filename: {filename}
        Media type: pdf
        
        First page text content:
        {first_page_text}
        
        {"Tables were extracted and saved for further analysis." if csv_path else "No tables were found or extracted."}
        
        Please provide a structured analysis including:
        - description: High-level description of the PDF content
        - detected_objects: Key entities, topics, or elements mentioned
        - textual_content: Summary of the text content (first 1000 chars)
        - statistical_info: Any numerical data, counts, or measurements mentioned
        - possible_analytical_uses: How this PDF could be used in data analysis
        - has_tables: {should_extract}
        - extract_all_tables: {should_extract}
        """

        result = await agent.run(prompt)

        if not isinstance(result.output, dict):
            print(
                f"ERROR: Agent output is not a dict for {file_path}. Got {type(result.output)}")
            return None

        # Ensure proper values for PDF-specific fields
        output_dict = result.output.copy()
        output_dict['filename'] = filename
        output_dict['media_type'] = 'pdf'
        output_dict['textual_content'] = first_page_text[:1000]  # Limit size
        output_dict['has_tables'] = should_extract
        output_dict['extract_all_tables'] = should_extract

        try:
            metadata = MediaMetadata(**output_dict)
            print(f"Successfully created MediaMetadata for PDF {file_path}")
            return metadata
        except Exception as serialize_error:
            print(
                f"SERIALIZATION ERROR for PDF {file_path}: {serialize_error}")
            print(f"Raw output was: {output_dict}")
            return None

    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
