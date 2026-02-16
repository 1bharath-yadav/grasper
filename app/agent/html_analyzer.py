import json
import asyncio
from typing import Dict, Any, List, Optional
import logging
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
import re
import os

logger = logging.getLogger(__name__)

# Global crawler instance for reuse
_crawler_instance = None


async def get_crawler():
    """Get or create a global crawler instance"""
    global _crawler_instance
    if not _crawler_instance:
        _crawler_instance = AsyncWebCrawler(verbose=True)
        await _crawler_instance.astart()
    return _crawler_instance


async def close_crawler():
    """Close the global crawler instance"""
    global _crawler_instance
    if _crawler_instance:
        await _crawler_instance.aclose()
        _crawler_instance = None


async def analyze_html(url: str, data_analyst_input: str, content: str = None) -> Dict[str, Any]:
    """
    Analyze web content using Crawl4AI and answer user questions based on data_analyst_input

    Args:
        url: The URL to crawl and analyze
        data_analyst_input: User's questions/queries about the web content
        content: Optional pre-fetched HTML content (will be ignored, we'll fetch fresh)
    """
    try:
        # Get crawler instance
        crawler = await get_crawler()

        # Create extraction strategy based on user input
        extraction_strategy = create_extraction_strategy(data_analyst_input)

        # Crawl the webpage with extraction strategy
        result = await crawler.arun(
            url=url,
            extraction_strategy=extraction_strategy,
            chunking_strategy=RegexChunking(),
            bypass_cache=True,
            process_iframes=True,
            remove_overlay_elements=True
        )

        if not result.success:
            return {
                "success": False,
                "error": f"Failed to crawl URL: {result.error_message}",
                "url": url,
                "data_analyst_input": data_analyst_input
            }

        # Extract structured information
        extracted_content = process_extraction_result(
            result, data_analyst_input)

        # Generate answer based on extracted content and user input
        answer = await generate_answer(extracted_content, data_analyst_input, result)

        return {
            "success": True,
            "url": url,
            "data_analyst_input": data_analyst_input,
            "extracted_content": extracted_content,
            "answer": answer,
            "page_title": result.metadata.get("title", ""),
            "page_description": result.metadata.get("description", ""),
            "links_found": len(result.links.get("internal", [])) + len(result.links.get("external", [])),
            "media_found": len(result.media.get("images", [])) + len(result.media.get("videos", [])),
            "word_count": len(result.cleaned_html.split()) if result.cleaned_html else 0
        }

    except Exception as e:
        logger.error(f"Error analyzing HTML with Crawl4AI: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "data_analyst_input": data_analyst_input
        }


def create_extraction_strategy(data_analyst_input: str) -> LLMExtractionStrategy:
    """
    Create an LLM extraction strategy based on user input
    """
    # Create a comprehensive prompt that includes the user's question
    extraction_prompt = f"""
    Based on the following user question/request: "{data_analyst_input}"

    Please extract and analyze the relevant information from this webpage. Focus on:

    1. Direct answers to the user's question
    2. Related data, statistics, numbers, or facts
    3. Tables, lists, or structured data that might be relevant
    4. Key information that addresses the user's needs
    5. Any specific elements mentioned in the user's question (like companies, products, dates, etc.)
    6. provide output exactly stated in user question/request

    If you cannot find relevant information, clearly state what information is not available.
    """

    # Check if API key is available
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    return LLMExtractionStrategy(
        provider="google/gemini-2.5-flash",
        api_token=gemini_api_key,
        instruction=extraction_prompt,
        schema={
            "type": "object",
            "properties": {
                "direct_answer": {"type": "string"},
                "relevant_data": {"type": "array", "items": {"type": "string"}},
                "tables_data": {"type": "array", "items": {"type": "string"}},
                "key_information": {"type": "array", "items": {"type": "string"}},
                "specific_elements": {"type": "array", "items": {"type": "string"}},
                "additional_context": {"type": "string"}
            }
        }
    )


def process_extraction_result(result, data_analyst_input: str) -> Dict[str, Any]:
    """
    Process the crawl result and extract relevant information
    """
    try:
        extracted_content = {
            "title": result.metadata.get("title", ""),
            "description": result.metadata.get("description", ""),
            "text_content": result.cleaned_html,
            "markdown_content": result.markdown,
            "links": {
                "internal": result.links.get("internal", []),
                "external": result.links.get("external", [])
            },
            "media": {
                "images": result.media.get("images", []),
                "videos": result.media.get("videos", []),
                "audios": result.media.get("audios", [])
            },
            "extracted_data": None
        }

        # If LLM extraction was used, parse the result
        if hasattr(result, 'extracted_content') and result.extracted_content:
            try:
                if isinstance(result.extracted_content, str):
                    extracted_content["extracted_data"] = json.loads(
                        result.extracted_content)
                else:
                    extracted_content["extracted_data"] = result.extracted_content
            except json.JSONDecodeError:
                extracted_content["extracted_data"] = {
                    "raw_extraction": result.extracted_content}

        return extracted_content

    except Exception as e:
        logger.error(f"Error processing extraction result: {str(e)}")
        return {
            "title": "",
            "description": "",
            "text_content": "",
            "markdown_content": "",
            "links": {"internal": [], "external": []},
            "media": {"images": [], "videos": [], "audios": []},
            "extracted_data": None,
            "error": str(e)
        }


async def generate_answer(extracted_content: Dict[str, Any],
                          data_analyst_input: str, result) -> str:
    """
    Generate a comprehensive answer based on extracted content and user input
    """
    try:
        # If we have LLM extracted data, use it primarily
        if extracted_content.get("extracted_data"):
            extracted_data = extracted_content["extracted_data"]

            answer_parts = []

            if extracted_data.get("direct_answer"):
                answer_parts.append(
                    f"Direct Answer: {extracted_data['direct_answer']}")

            if extracted_data.get("relevant_data"):
                answer_parts.append("\nRelevant Data:")
                for data_point in extracted_data["relevant_data"]:
                    answer_parts.append(f"• {data_point}")

            if extracted_data.get("key_information"):
                answer_parts.append("\nKey Information:")
                for info in extracted_data["key_information"]:
                    answer_parts.append(f"• {info}")

            if extracted_data.get("tables_data"):
                answer_parts.append("\nTable Data:")
                for table_info in extracted_data["tables_data"]:
                    answer_parts.append(f"• {table_info}")

            if extracted_data.get("specific_elements"):
                answer_parts.append("\nSpecific Elements Found:")
                for element in extracted_data["specific_elements"]:
                    answer_parts.append(f"• {element}")

            if extracted_data.get("additional_context"):
                answer_parts.append(
                    f"\nAdditional Context: {extracted_data['additional_context']}")

            if answer_parts:
                return "\n".join(answer_parts)

        # Fallback: Manual content analysis
        return _manual_content_analysis(extracted_content, data_analyst_input)

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return _manual_content_analysis(extracted_content, data_analyst_input)


def _manual_content_analysis(extracted_content: Dict[str, Any],
                             data_analyst_input: str) -> str:
    """
    Fallback manual content analysis when LLM extraction is not available
    """
    try:
        answer_parts = []

        # Basic page information
        if extracted_content.get("title"):
            answer_parts.append(
                f"Page Title: {extracted_content['title']}")

        if extracted_content.get("description"):
            answer_parts.append(
                f"Page Description: {extracted_content['description']}")

        # Content analysis based on user input
        input_lower = data_analyst_input.lower()
        text_content = extracted_content.get("text_content", "")

        # Extract keywords from user input
        keywords = re.findall(r'\b\w{3,}\b', input_lower)
        keywords = [k for k in keywords if k not in [
            'what', 'how', 'when', 'where', 'why', 'the', 'and', 'for']]

        # Find relevant sentences
        if text_content and keywords:
            sentences = re.split(r'[.!?]+', text_content)
            relevant_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    relevant_sentences.append(sentence.strip())
                    if len(relevant_sentences) >= 5:
                        break

            if relevant_sentences:
                answer_parts.append("\nRelevant Information Found:")
                for sentence in relevant_sentences:
                    if len(sentence) > 20:  # Filter out very short sentences
                        answer_parts.append(f"• {sentence}")

        # Links information
        internal_links = extracted_content.get(
            "links", {}).get("internal", [])
        external_links = extracted_content.get(
            "links", {}).get("external", [])

        if internal_links or external_links:
            answer_parts.append(
                f"\nLinks Found: {len(internal_links)} internal, {len(external_links)} external")

        # Media information
        media = extracted_content.get("media", {})
        images = media.get("images", [])
        videos = media.get("videos", [])

        if images or videos:
            answer_parts.append(
                f"\nMedia Found: {len(images)} images, {len(videos)} videos")

        if not answer_parts:
            answer_parts.append(
                "I was able to access the webpage, but couldn't find specific information related to your query. The page may not contain the information you're looking for, or it might be structured in a way that makes extraction difficult.")

        return "\n".join(answer_parts)

    except Exception as e:
        logger.error(f"Error in manual content analysis: {str(e)}")
        return f"I encountered an error while analyzing the webpage content: {str(e)}"


async def close(self):
    """Close the crawler session"""
    if self.crawler:
        await self.crawler.aclose()
        self.crawler = None
