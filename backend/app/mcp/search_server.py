import urllib.request
import urllib.parse
import json
import re
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("Tutor Search Assistant")

@mcp.tool()
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for a given topic and return the summary of the best matching article.
    Use this when you need comprehensive background knowledge, definitions, or historical context.
    """
    print(f"[MCP SERVER] Tool called: search_wikipedia with query: '{query}'")
    try:
        # Step 1: Search for matching articles
        search_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + urllib.parse.quote_plus(query) + "&format=json"
        req = urllib.request.Request(
            search_url,
            headers={'User-Agent': 'TutorLearnerDebateResearch/2.0 (student-portfolio-project; contact: franciellemdn@github.com)'}
        )
        
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return f"No Wikipedia articles found matching '{query}'."
            
        best_match = search_results[0]["title"]
        
        # Step 2: Retrieve extract/summary for the best matching article title
        summary_url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&exsentences=4&titles=" + urllib.parse.quote_plus(best_match) + "&format=json"
        req = urllib.request.Request(
            summary_url,
            headers={'User-Agent': 'TutorLearnerDebateResearch/2.0 (student-portfolio-project; contact: franciellemdn@github.com)'}
        )
        
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return f"Could not retrieve summary for article '{best_match}'."
            
        page_id = list(pages.keys())[0]
        extract = pages[page_id].get("extract", "").strip()
        
        if not extract:
            return f"No summary extract available for '{best_match}'."
            
        return f"Article: {best_match}\n\nSummary:\n{extract}"
    except Exception as e:
        return f"Failed to retrieve Wikipedia article: {str(e)}"

@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the live web using DuckDuckGo and return top result snippets.
    Use this for up-to-date facts, news, or general web inquiries.
    """
    print(f"[MCP SERVER] Tool called: web_search with query: '{query}'")
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        req = urllib.request.Request(
            url,
            headers={
                # Use a standard browser User-Agent to avoid blocks
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        )
        
        # Fetch search results
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all search result snippets
        snippets = []
        result_elements = soup.find_all('div', class_='result')
        
        for el in result_elements[:4]:
            title_el = el.find('a', class_='result__url')
            snippet_el = el.find('a', class_='result__snippet')
            
            if title_el and snippet_el:
                title = title_el.get_text().strip()
                snippet = snippet_el.get_text().strip()
                snippets.append(f"Title: {title}\nSnippet: {snippet}")
                
        if not snippets:
            return f"No search results returned for query '{query}'."
            
        return "\n\n---\n\n".join(snippets)
    except Exception as e:
        return f"Web search failed: {str(e)}"

if __name__ == "__main__":
    mcp.run()
