import requests
from typing import List, Dict

class LLMOrchestrator:
    """Orchestrates interactions with the LLM service for search result processing."""
    
    DEFAULT_MODEL = "mistral"
    API_ENDPOINT = "http://localhost:11434/api/generate"
    CONTENT_PREVIEW_LENGTH = 1000
    REQUEST_TIMEOUT = 10

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the LLM orchestrator.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model = model_name
        self.api_url = self.API_ENDPOINT

    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate an LLM response for the search results.
        
        Args:
            query: User's search query
            search_results: List of search result dictionaries
            
        Returns:
            str: Generated response or fallback formatted results
        """
        try:
            formatted_results = self._format_results_for_prompt(search_results)
            prompt = self._construct_prompt(query, formatted_results)
            
            return self._get_llm_response(prompt)
                
        except requests.exceptions.ConnectionError:
            self._print_ollama_setup_instructions()
            return self._format_results_without_llm(search_results)
        except Exception as err:
            print(f"\nError generating LLM response: {err}")
            return self._format_results_without_llm(search_results)

    def _format_results_for_prompt(self, search_results: List[Dict]) -> List[str]:
        """Format search results for inclusion in the LLM prompt."""
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            metadata = result['metadata']
            if not metadata:
                continue
                
            details = [f"\n{idx}. File: {metadata['path']}"]
            self._add_metadata_details(details, metadata)
            formatted_results.append("\n".join(details))
            
        return formatted_results

    def _add_metadata_details(self, details: List[str], metadata: Dict):
        """Add metadata details to the formatted result."""
        if metadata.get('title'):
            details.append(f"Title: {metadata['title']}")
        if metadata.get('artist'):
            details.append(f"Artist: {metadata['artist']}")
        if metadata.get('duration'):
            details.append(f"Duration: {metadata['duration']:.2f} seconds")
        if metadata.get('dimensions'):
            details.append(f"Dimensions: {metadata['dimensions']}")
        if metadata.get('page_count'):
            details.append(f"Pages: {metadata['page_count']}")
        
        # Add content with clear separation
        if metadata.get('content'):
            truncated_content = metadata['content'][:1000].replace('\n', ' ').strip()
            if truncated_content:
                details.append(f"Content: {truncated_content}")

    def _construct_prompt(self, query: str, formatted_results: List[str]) -> str:
        """Construct the LLM prompt based on the formatted search results."""
        return f"""You are a helpful assistant analyzing search results from a file system. 
The user's query is: "{query}"

Here are the most relevant files and their details:
{chr(10).join(formatted_results)}

Please provide a clear and concise response that:
1. Directly answers the user's query using the available information
2. Cites specific files and their content when relevant
3. Highlights exact matches or relevant excerpts
4. Mentions if the requested information isn't found in the results

Response format:
- Start with a direct answer to the query
- List relevant files with their key information
- Include specific quotes or data points that support the answer
- End with any necessary caveats or additional context

Keep your response focused and relevant to the query."""

    def _get_llm_response(self, prompt: str) -> str:
        """Get the LLM response for the given prompt."""
        # Call Ollama API
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=self.REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return self._format_results_without_llm(search_results)

    def _format_results_without_llm(self, search_results: List[Dict]) -> str:
        """Format search results when LLM is unavailable"""
        if not search_results:
            return "No matching files found."
            
        output = ["🔍 Search Results", "─────────────"]
        
        for idx, result in enumerate(search_results, 1):
            metadata = result['metadata']
            if not metadata:
                continue
                
            # Add file info with path
            output.append(f"\n{idx}. {metadata['path']}")
            
            # Create a compact metadata section
            details = []
            
            # Basic file info
            if metadata.get('size'):
                size_bytes = metadata['size']
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_bytes < 1024:
                        details.append(f"Size: {size_bytes:.1f}{unit}")
                        break
                    size_bytes /= 1024
            
            # Media-specific metadata (compact format)
            meta_parts = []
            if metadata.get('title'):
                meta_parts.append(f"Title: {metadata['title']}")
            if metadata.get('artist'):
                meta_parts.append(f"Artist: {metadata['artist']}")
            if metadata.get('duration'):
                meta_parts.append(f"Duration: {metadata['duration']:.1f}s")
            if metadata.get('dimensions'):
                meta_parts.append(f"Dim: {metadata['dimensions']}")
            if metadata.get('page_count'):
                meta_parts.append(f"Pages: {metadata['page_count']}")
            
            if meta_parts:
                details.append(" | ".join(meta_parts))
                
            # Add content preview if available (single line, shorter)
            if metadata.get('content'):
                content = metadata['content'][:100].replace('\n', ' ').strip()
                if content:
                    details.append(f"Preview: \"{content}...\"")
            
            # Add all details with minimal indentation
            output.extend(f"  • {detail}" for detail in details)
            
        return "\n".join(output) 