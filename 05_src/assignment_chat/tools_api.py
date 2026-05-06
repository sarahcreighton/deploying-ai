import requests
import xml.etree.ElementTree as ET
import time
import pandas as pd
from langchain_core.tools import tool


# -----------------------------
# Helper: Format APA Citation
# -----------------------------
def _format_pubmed_citation(paper: dict) -> str:
    """ Formats pubmed result in simplified APA format """
    authors = paper.get('authors', 'Unknown')
    year = paper.get('year', 'n.d.')
    title = paper.get('title', 'Unknown Title')
    journal = paper.get('journal', 'Unknown Journal')
    pmid = paper.get('pmid', '')
    pmid_link = f" [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
    return f"{authors} ({year}). [{title}]({pmid_link}) *{journal}*."

# -----------------------------
# Step 1: Search PubMed IDs
# -----------------------------
def _search_pubmed_ids(query: str, max_results: int = 20):
    """ Search PubMed IDs (pmid) """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml",
        "sort": "relevance",
        "email": "your_email@example.com" # for API identification
    }
    r = requests.get(url, params=params)
    r.raise_for_status() # check for http errors
    root = ET.fromstring(r.content)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    return pmids


# ------------------------------------
# Step 2a: Fetch ONLY Metadata (Fast)
# ------------------------------------
def _get_pubmed_metadata(pmids):
    """ Fetch only metadata (Title, Authors, Year, Journal, PMID) for speed. """
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "sort": "relevance",
        "email": "your_email@example.com" # for API identification
    }
    r = requests.get(url, params=params)
    r.raise_for_status() # check for http errors
    root = ET.fromstring(r.content)

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
            author_list_node = article_node.find(".//AuthorList")
            if author_list_node is not None:
                for author in author_list_node.findall(".//Author"):
                    last_name = author.findtext(".//LastName", "")
                    fore_name = author.findtext(".//ForeName", "")
                    if last_name:
                        authors_list.append(f"{last_name} {fore_name}" if fore_name else last_name)
            authors_str = ", ".join(authors_list) if authors_list else "Unknown"

            # Year
            pub_year = ""
            pub_date = article_node.find(".//JournalIssue//PubDate//Year")
            if pub_date is not None:
                pub_year = pub_date.text
            if not pub_year:
                pub_data = article.find(".//PubData//ArticleDate//Year")
                if pub_data is not None:
                    pub_year = pub_data.text

            # Combine
            papers.append({
                "title": title,
                "authors": authors_str,
                "year": pub_year,
                "journal": journal,
                "pmid": article.findtext(".//PMID"),
                "source": "pubmed"
            })
        except Exception as e:
            print(f"Error processing PMID {article.findtext('.//PMID')}: {e}")
            continue
    return papers


# ------------------------------------
# Step 2b: Fetch Abstract (On-Demand)
# ------------------------------------
def _get_pubmed_abstract(pmid: str) -> str:
    """ Fetch the abstract for a single PMID. """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    
    abstract_text = ""
    abstract_node = root.find(".//Abstract")
    if abstract_node is not None:
        abstract_text = abstract_node.findtext(".//AbstractText", "").strip()    
    return abstract_text


# --------------------------------------
# Step 3a: Tool for Citation List (Fast)
# --------------------------------------
@tool
def search_pubmed_live(query: str = "vision AND psychophysics", max_results: int = 5) -> str:
    """
    Use this tool when the user asks you to search for RECENT papers on a 
    topic that may not be contained in the local chroma database.

    Examples:   "Find preprints on topic X." or 
                "What's the latest paper from researcher Y?" 

    Returns a formatted list of citations (Title, Authors, Year, Journal).
    Does NOT include abstracts by default to save time.
    """
    pmids = _search_pubmed_ids(query, max_results)
    time.sleep(0.5)  # be polite to API
    metadata = _get_pubmed_metadata(pmids)
    citations = [_format_pubmed_citation(p) for p in metadata]
    return "\n\n".join(citations)


# -------------------------------------------------
# Step 3b: Tool to Get Abstract for Specific Paper
# -------------------------------------------------
@tool
def return_pubmed_abstract(pmid: str) -> str:
    """
    Use this tool to get the full abstract for a 
    specific paper when the user asks for details.
    Returns as formatted text.
    """
    abstract = _get_pubmed_abstract(pmid)
    return abstract