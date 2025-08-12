import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from bs4 import BeautifulSoup, Comment, Tag
import json
import re
import logfire
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logfire.configure()

# Load API key
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Pydantic models for structured responses


class HTMLElement(BaseModel):
    tag: str
    id: str = ""
    classes: List[str] = []
    text_content: str = ""
    attributes: Dict[str, str] = {}
    element_type: str = ""  # form, link, heading, content, etc.


class WebsiteStructure(BaseModel):
    title: str = ""
    description: str = ""
    headings: List[Dict[str, str]] = []
    forms: List[Dict[str, Any]] = []
    links: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []
    main_content: str = ""
    meta_info: Dict[str, str] = {}


class AnalysisResponse(BaseModel):
    answer: str = Field(description="Direct answer to the user's question")
    confidence: float = Field(
        description="Confidence score between 0-1", ge=0, le=1)
    relevant_elements: List[str] = Field(
        description="HTML elements that support this answer")
    reasoning: str = Field(
        description="Explanation of how the answer was derived")
    additional_context: Optional[str] = Field(
        default=None, description="Any additional relevant information")


# Configure Google provider and model
provider = GoogleProvider(api_key=GEMINI_KEY)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# Initialize Pydantic AI agent
html_analyzer_agent = Agent(
    model,
    system_prompt="""
    You are an expert HTML analyzer. Your job is to analyze website content and answer user questions accurately.
    
    Guidelines:
    1. Base your answers strictly on the provided HTML content
    2. Reference specific HTML elements when possible
    3. Provide confidence scores based on how clearly the HTML supports your answer
    4. If information is not available in the HTML, state this clearly
    5. Focus on semantic meaning, not just text matching
    6. Consider the context and structure of the website
    
    Always provide:
    - A direct, clear answer
    - Confidence score (0.0-1.0)
    - Relevant HTML elements that support your answer
    - Clear reasoning for your conclusion
    """
)


def load_html_file(html_path: Union[str, Path]) -> str:
    """Load HTML content from file path."""
    try:
        html_path = Path(html_path)
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")

        with open(html_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(html_path, 'r', encoding='latin-1') as file:
            return file.read()


def prune_html(html_content: str) -> str:
    """
    Prune HTML content to remove unnecessary elements while preserving semantic structure.
    Based on the paper's HTML pruning approach.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove comments, scripts, styles, and other non-content elements
    for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
        element.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Keep only essential attributes for key elements
    essential_attrs = {
        'input': ['type', 'name', 'id', 'placeholder', 'value', 'required'],
        'form': ['action', 'method', 'name', 'id'],
        'a': ['href', 'title', 'id'],
        'img': ['src', 'alt', 'title', 'id'],
        'button': ['type', 'name', 'id'],
        'select': ['name', 'id'],
        'textarea': ['name', 'id', 'placeholder'],
        'div': ['id', 'class'],
        'span': ['id', 'class'],
        'h1': ['id', 'class'], 'h2': ['id', 'class'], 'h3': ['id', 'class'],
        'h4': ['id', 'class'], 'h5': ['id', 'class'], 'h6': ['id', 'class']
    }

    # Clean up attributes
    for element in soup.find_all():
        if isinstance(element, Tag):  # Only process Tag elements, not strings
            if element.name in essential_attrs:
                # Keep only essential attributes
                attrs_to_keep = essential_attrs[element.name]
                element.attrs = {k: v for k, v in element.attrs.items()
                                 if k in attrs_to_keep}
            else:
                # For other tags, keep only id and class
                element.attrs = {k: v for k, v in element.attrs.items() if k in [
                    'id', 'class']}

    return str(soup)


def extract_website_structure(html_content: str) -> WebsiteStructure:
    """
    Extract structured information from HTML content.
    Implements the parser-processed approach from the paper.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title
    title_tag = soup.find('title')
    title = title_tag.get_text().strip() if title_tag else ""

    # Extract meta description
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    description = desc_tag.get(
        'content', "") if desc_tag and isinstance(desc_tag, Tag) else ""

    # Extract headings
    headings = []
    for i in range(1, 7):
        for heading in soup.find_all(f'h{i}'):
            if isinstance(heading, Tag):
                headings.append({
                    'level': str(i),  # Convert to string for Pydantic model
                    'text': heading.get_text().strip(),
                    'id': heading.get('id', ''),
                    'tag': f'h{i}'
                })

    # Extract forms
    forms = []
    for form in soup.find_all('form'):
        if isinstance(form, Tag):
            form_data = {
                'id': form.get('id', ''),
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'inputs': []
            }

            for input_elem in form.find_all(['input', 'select', 'textarea']):
                if isinstance(input_elem, Tag):
                    input_data = {
                        'tag': input_elem.name,
                        'type': input_elem.get('type', ''),
                        'name': input_elem.get('name', ''),
                        'id': input_elem.get('id', ''),
                        'placeholder': input_elem.get('placeholder', ''),
                        'value': input_elem.get('value', '')
                    }
                    form_data['inputs'].append(input_data)

            forms.append(form_data)

    # Extract links
    links = []
    for link in soup.find_all('a', href=True):
        if isinstance(link, Tag):
            links.append({
                'text': link.get_text().strip(),
                'href': link.get('href', ''),
                'title': link.get('title', ''),
                'id': link.get('id', '')
            })

    # Extract images
    images = []
    for img in soup.find_all('img'):
        if isinstance(img, Tag):
            images.append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'id': img.get('id', '')
            })

    # Extract main content (remove nav, header, footer, sidebar)
    content_soup = BeautifulSoup(str(soup), 'html.parser')
    for element in content_soup(['nav', 'header', 'footer', 'aside']):
        element.decompose()

    main_content = content_soup.get_text().strip()
    # Clean up whitespace
    main_content = re.sub(r'\s+', ' ', main_content)[:2000]  # Limit length

    # Extract meta information
    meta_info = {}
    for meta in soup.find_all('meta'):
        if isinstance(meta, Tag):
            name = meta.get('name') or meta.get('property', '')
            content = meta.get('content', '')
            if name and content:
                meta_info[str(name)] = str(content)

    return WebsiteStructure(
        title=title,
        description=str(description) if description else "",
        headings=headings,
        forms=forms,
        links=links,
        images=images,
        main_content=main_content,
        meta_info=meta_info
    )


def create_analysis_prompt(structure: WebsiteStructure, questions: List[str]) -> str:
    """
    Create a comprehensive prompt for analysis.
    Based on the paper's prompt design framework.
    """
    # Convert structure to JSON for the prompt
    structure_json = structure.model_dump()

    # Format questions
    questions_text = "\n".join(
        [f"Q{i+1}: {q}" for i, q in enumerate(questions)])

    prompt = f"""
    WEBSITE ANALYSIS TASK
    
    You are analyzing a website with the following structure and content:
    
    WEBSITE INFORMATION:
    - Title: {structure.title}
    - Description: {structure.description}
    - Number of headings: {len(structure.headings)}
    - Number of forms: {len(structure.forms)}
    - Number of links: {len(structure.links)}
    - Number of images: {len(structure.images)}
    
    DETAILED STRUCTURE:
    {json.dumps(structure_json, indent=2)}
    
    QUESTIONS TO ANSWER:
    {questions_text}
    
    INSTRUCTIONS:
    1. Answer each question based strictly on the provided website structure
    2. Reference specific elements (by ID, text content, or tag type) that support your answer
    3. If information is not available, clearly state this
    4. Provide confidence scores based on how well the HTML supports your conclusions
    5. Give detailed reasoning for your answers
    
    For multiple questions, answer them in order (Q1, Q2, etc.) but provide a single consolidated response.
    """

    return prompt


async def analyze_html(html_path: Union[str, Path], questions: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Main function to analyze HTML content and answer questions.

    Args:
        html_path: Path to the HTML file
        questions: Single question string or list of questions

    Returns:
        AnalysisResponse: Structured response with answers and analysis
    """
    try:
        # Ensure questions is a list
        if isinstance(questions, str):
            questions = [questions]

        # Load HTML content
        print(f"Loading HTML from: {html_path}")
        html_content = load_html_file(html_path)

        # Prune HTML to remove unnecessary content
        print("Processing HTML content...")
        pruned_html = prune_html(html_content)

        # Extract structured information
        website_structure = extract_website_structure(pruned_html)
        print(f"Extracted structure: {len(website_structure.headings)} headings, "
              f"{len(website_structure.forms)} forms, {len(website_structure.links)} links")

        # Create analysis prompt
        prompt = create_analysis_prompt(website_structure, questions)

        # Get AI analysis
        print("Analyzing with AI...")
        response = await html_analyzer_agent.run(prompt)

        # Since we're not using structured output, create a simple response
        return {
            "answer": str(response),
            "confidence": 0.8,  # Default confidence
            "relevant_elements": ["HTML structure analyzed"],
            "reasoning": "Analysis based on HTML content structure",
            "additional_context": f"Analyzed {len(website_structure.headings)} headings, {len(website_structure.forms)} forms, {len(website_structure.links)} links"
        }

    except Exception as e:
        # Return error response
        return {
            "answer": f"Error analyzing HTML: {str(e)}",
            "confidence": 0.0,
            "relevant_elements": [],
            "reasoning": f"An error occurred during analysis: {str(e)}",
            "additional_context": "Please check the HTML file path and content."
        }

# Synchronous wrapper for easier use


def analyze_html_sync(html_path: Union[str, Path], questions: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Synchronous wrapper for the analyze_html function.
    """
    return asyncio.run(analyze_html(html_path, questions))


def analyze_html_default(html_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Default HTML analysis without specific questions.
    Returns a dictionary with general website information suitable for data analysis.

    Args:
        html_path: Path to the HTML file

    Returns:
        Dict containing website structure and basic analysis
    """
    try:
        # Load and process HTML
        html_content = load_html_file(html_path)
        pruned_html = prune_html(html_content)
        structure = extract_website_structure(pruned_html)

        # Convert to dictionary for JSON serialization
        result = {
            "title": structure.title,
            "description": structure.description,
            "headings_count": len(structure.headings),
            "headings": structure.headings[:10],  # Limit to first 10
            "forms_count": len(structure.forms),
            "forms": structure.forms,
            "links_count": len(structure.links),
            "links": structure.links[:20],  # Limit to first 20
            "images_count": len(structure.images),
            "images": structure.images[:10],  # Limit to first 10
            "main_content_preview": structure.main_content[:500] if structure.main_content else "",
            "meta_info": structure.meta_info,
            "analysis_type": "html_structure"
        }

        return result

    except Exception as e:
        return {
            "error": str(e),
            "analysis_type": "html_structure",
            "title": "",
            "description": "",
            "headings_count": 0,
            "forms_count": 0,
            "links_count": 0,
            "images_count": 0
        }

# Example usage and testing function


async def test_analyzer():
    """Test function to demonstrate usage."""
    # Example HTML content for testing
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample E-commerce Site</title>
        <meta name="description" content="Best online store for electronics">
    </head>
    <body>
        <header>
            <h1>TechStore</h1>
            <nav>
                <a href="/products">Products</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
            </nav>
        </header>
        <main>
            <h2>Featured Products</h2>
            <div class="product">
                <h3>Smartphone Pro</h3>
                <p>Latest model with advanced features</p>
                <img src="phone.jpg" alt="Smartphone Pro">
                <form id="purchase-form" method="post" action="/buy">
                    <input type="hidden" name="product_id" value="123">
                    <input type="number" name="quantity" placeholder="Quantity" min="1" value="1">
                    <button type="submit">Add to Cart</button>
                </form>
            </div>
        </main>
        <footer>
            <p>&copy; 2024 TechStore</p>
        </footer>
    </body>
    </html>
    """

    # Save test HTML to file
    test_file = Path("test_website.html")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_html)

    # Test questions
    test_questions = [
        "What is the name of this website?",
        "What products are featured on the homepage?",
        "How can users purchase products?",
        "What navigation options are available?"
    ]

    # Analyze
    result = await analyze_html(test_file, test_questions)

    print("=== ANALYSIS RESULT ===")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Relevant Elements: {result['relevant_elements']}")
    print(f"Reasoning: {result['reasoning']}")
    if result.get('additional_context'):
        print(f"Additional Context: {result['additional_context']}")

    # Clean up
    test_file.unlink()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_analyzer())
