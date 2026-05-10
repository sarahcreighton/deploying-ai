import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict
from langchain_core.tools import tool

# -----------------------------
# TOOL DEFINITIONS
# -----------------------------
# NOTE: search_pubmed() would be better as a structured tool
# Significant prompting currently required to force chatbot 
# to return as requested by _format_papers_for_chat(). Works 
# for now, but fragile. Refer to notes.ipynb for a draft

@tool
def search_pubmed(query: str = "vision AND psychophysics", max_results: int = 5) -> str:
    """
    Use this tool when the user asks you to search for RECENT papers on a 
    topic that may not be contained in the local chroma database.
    Examples:   "Find preprints on topic X." or 
                "What's the latest paper from researcher Y?" 
    Returns:
        formatted list of citations (Title, Authors, Year, Journal)
        abstract (optional)
    """
    pmids = _search_pubmed_ids(query, max_results) # get IDs
    time.sleep(0.5)  # be polite to API
    papers = _fetch_pubmed_data(pmids, include_abstract=True) # get metadata
    response = _format_papers_for_chat(papers, query_context=query)
    return response


# -----------------------------
# Step 1: Execute PubMed Query
# -----------------------------
def _search_pubmed_ids(query: str, max_results: int = 20) -> List[str]:
    """ Search PubMed IDs (pmid) via eSearch """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml",
        "sort": "relevance",
        "email": "your_email@example.com"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        pmids = [id_elem.text for id_elem in root.findall(".//Id")]
        return pmids
    except requests.RequestException as e:
        print(f"Search API Network Error: {e}")
        return []
    except ET.ParseError as e: 
        print(f"XML Parsing Error: Received malformed data from API. {e}")
        return []


# ------------------------------------
# Step 2: Parse XML Data
# ------------------------------------
def _parse_pubmed_xml(xml_content: str, include_abstract: bool = True) -> List[Dict]:
    """
    Parses PubMed XML data into list of dictionaries.
    Args:
        xml_content: Raw XML string from PubMed API.
        include_abstract: If True, attempts to extract AbstractText.
    Returns:
        List of dictionaries with paper details.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return []

    papers = []
    for article in root.findall(".//PubmedArticle"):
        try:
            article_node = article.find(".//Article")
            if article_node is None:
                continue

            # Article Title & Journal
            title = article.findtext(".//ArticleTitle", "").strip()
            journal = article_node.findtext(".//Journal//Title", "")
            
            # Authors
            authors_list = []
            author_list_node = article.find(".//AuthorList")
            if author_list_node is not None:
                for author in author_list_node.findall(".//Author"):
                    last = author.findtext(".//LastName", "")
                    first = author.findtext(".//ForeName", "")
                    if last:
                        authors_list.append(f"{last} {first}" if first else last)
            authors_str = ", ".join(authors_list) if authors_list else "Unknown"

            # Year
            year = ""
            pub_date = article_node.find(".//JournalIssue//PubDate//Year")
            if pub_date is not None:
                year = pub_date.text
            else:
                pub_data = article.find(".//PubData//ArticleDate//Year")
                if pub_data is not None:
                    year = pub_data.text
            
            # PMID
            pmid = article.findtext(".//PMID", "")
            pmid_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ""

            # Abstract (Optional)
            abstract_text = ""
            if include_abstract:
                abstract_node = article.find(".//Abstract")
                if abstract_node is not None:
                    # Get all AbstractText elements and join them
                    abstract_texts = abstract_node.findall(".//AbstractText")
                    if abstract_texts:
                        # Join with newlines to preserve paragraph structure
                        abstract_text = "\n".join([t.text for t in abstract_texts if t.text])
                    else:
                        abstract_text = abstract_node.findtext(".//AbstractText", "").strip()

            papers.append({
                "pmid": pmid,
                "title": title,
                "authors": authors_str,
                "year": year,
                "journal": journal,
                "abstract": abstract_text,
                "citation": f"{authors_str} ({year}). [{title}]({pmid_link}). *{journal}*.",
                "source": "pubmed"
            })
            
        except Exception as e:
            # Log error but continue to next paper
            print(f"Error parsing paper {article.findtext('.//PMID')}: {e}")
            continue         
    return papers


# ------------------------------------
# Step 3: Collect Article Metadata
# ------------------------------------
def _fetch_pubmed_data(pmids: List[str], include_abstract: bool = True) -> List[Dict]:
    """
    Fetches data from PubMed efetch API.
    """
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract" if include_abstract else "xml", # rettype='abstract' is faster if you only want abstracts
        "email": "your_email@example.com" # Replace with your email
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return _parse_pubmed_xml(r.text, include_abstract=include_abstract)
    except requests.RequestException as e:
        print(f"API Error: {e}")
        return []


# ------------------------------------
# Step 4: Format PubMed Citations
# ------------------------------------
def _format_pubmed_citation(paper: Dict) -> str:
    """ Formats pubmed result in simplified APA format """
    authors = paper.get('authors', 'Unknown')
    year = paper.get('year', 'n.d.')
    title = paper.get('title', 'Unknown Title')
    journal = paper.get('journal', 'Unknown Journal')
    pmid = paper.get('pmid', '')
    pmid_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    return f"{authors} ({year}). [{title}]({pmid_link}) *{journal}*. PMID: {pmid}"

def _format_papers_for_chat(papers: list, query_context: str = "") -> str:
    """
    Converts a list of paper dictionaries into a human-readable, chat-friendly string
    Args:
        papers (list): List of dictionaries containing paper details.
        query_context (str): Optional context (e.g., the search query) to include in header
    Returns:
        str: Formatted markdown string.
    """
    if not papers:
        return "No papers found."
    
    formatted_blocks = []

    for i, paper in enumerate(papers, 1):
        abstract = paper.get("abstract", "")
        if len(abstract) > 400: # truncate if long
            abstract_display = abstract[:400] + "..."
        else:
            abstract_display = abstract

        # clean block for each paper
        block = (
            f"**{i}. {_format_pubmed_citation(paper)}**\n"
            f"*Abstract*: {abstract_display}"
        )
        formatted_blocks.append(block)

    # create the header
    if query_context:
        header = f"Here are the top {len(papers)} papers for your query: *'{query_context}'*"
    else:
        header - f"Here are the details for {len(papers)} papers:"
        
    return f"{header}\n\n" + "\n---\n\n".join(formatted_blocks)