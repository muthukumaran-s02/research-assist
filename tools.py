import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from logger import log_tool_call, logger
from config import MAX_RESULTS
import io

# Initialize DuckDuckGo tool
ddg_search = DuckDuckGoSearchRun()

@tool
def web_search(query: str):
    """Searches the web for information using DuckDuckGo."""
    logger.info(f"Searching for: {query}...")
    try:
        results = ddg_search.run(query)
        log_tool_call("web_search", {"query": query}, {"results": results})
        return results
    except Exception as e:
        logger.error(f"Error during search: {e}")
        log_tool_call("web_search", {"query": query}, {"error": str(e)})
        return f"Error during search: {e}"

@tool
def extract_document(url_or_path: str):
    """Extracts text content from a web URL or a local PDF file."""
    logger.info(f"Extracting: {url_or_path}...")
    try:
        content = ""
        # Handle local PDF
        if url_or_path.lower().endswith('.pdf'):
            if url_or_path.startswith('http'):
                # Download and parse
                response = requests.get(url_or_path, timeout=10)
                f = io.BytesIO(response.content)
                reader = PdfReader(f)
            else:
                # Local file
                reader = PdfReader(url_or_path)
            
            for page in reader.pages:
                content += page.extract_text() + "\n"
        
        # Handle URL
        else:
            response = requests.get(url_or_path, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove scripts and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            content = soup.get_text(separator=' ', strip=True)
        
        # Limit content for LLM context window safety
        preview = content[:2000] + "..." if len(content) > 2000 else content
        log_tool_call("extract_document", {"target": url_or_path}, {"preview": preview})
        return preview

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        log_tool_call("extract_document", {"target": url_or_path}, {"error": str(e)})
        return f"Error during extraction: {e}"

@tool
def web_fetch(url: str):
    """Fetches the text content of a given web URL."""
    logger.info(f"Fetching content from: {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        content = soup.get_text(separator=' ', strip=True)
        # Limit content for LLM context window safety
        preview = content[:3000] + "..." if len(content) > 3000 else content
        
        log_tool_call("web_fetch", {"url": url}, {"preview": preview})
        return preview
    except Exception as e:
        logger.error(f"Error during web fetch: {e}")
        log_tool_call("web_fetch", {"url": url}, {"error": str(e)})
        return f"Error fetching {url}: {e}"

@tool
def wiki_search(query: str):
    """Searches Wikipedia for a summary of a given topic."""
    logger.info(f"Searching Wikipedia for: {query}...")
    try:
        # Search for the most relevant page title
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()

        if not search_data['query']['search']:
            log_tool_call("wiki_search", {"query": query}, {"result": "No Wikipedia page found."})
            return "No relevant Wikipedia page found."

        page_title = search_data['query']['search'][0]['title']

        # Fetch the summary of the page
        summary_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": page_title,
            "format": "json"
        }
        summary_response = requests.get(search_url, params=summary_params, timeout=10)
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        pages = summary_data['query']['pages']
        page_id = next(iter(pages))
        summary = pages[page_id].get('extract', 'No summary available.')

        log_tool_call("wiki_search", {"query": query}, {"title": page_title, "summary": summary[:500] + "..."})
        return f"Wikipedia Summary for '{page_title}':\n{summary}"

    except Exception as e:
        logger.error(f"Error during Wikipedia search: {e}")
        log_tool_call("wiki_search", {"query": query}, {"error": str(e)})
        return f"Error searching Wikipedia: {e}"

tools = [wiki_search, extract_document, web_fetch]
