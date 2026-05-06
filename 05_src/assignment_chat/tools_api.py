import requests
import xml.etree.ElementTree as ET
import time
import pandas as pd
from langchain_core.tools import tool




# -----------------------------
# Step 1: Search PubMed IDs
# -----------------------------
def search_pubmed(query: str, max_results: int = 20):
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


# -----------------------------
# Step 2: Fetch Article Details
# -----------------------------
def fetch_pubmed_details(pmids):
    """ Fetch detailed information for a list of PMIDs. """
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
            # -- 1. Extract Title
            article_node = article.find(".//Article")
            if article_node is None:
                continue
            
            title_node = article.findtext(".//ArticleTitle")
            title = title_node.strip() if title_node else ""

            # -- 2. Extract Journal
            journal_node = article_node.find(".//Journal")
            journal_title = ""
            if journal_node is not None:
                journal_title = journal_node.findtext(".//Title")

            # -- 3. Extract Authors
            authors_list = []
            author_list_node = article_node.find(".//AuthorList")
            if author_list_node is not None:
                for author in author_list_node.findall(".//Author"):
                    last_name = author.findtext(".//LastName")
                    fore_name = author.findtext(".//ForeName")
                    if last_name:
                        author_str = f"{last_name} {fore_name}" if fore_name else last_name
                        authors_list.append(author_str)
            
            authors_str = ", ".join(authors_list) if authors_list else "Unknown"

            # -- 4. Extract Publication Date (ROBUST LOGIC)
            pub_year = None

            # Strategy A: Look in JournalIssue 
            journal_issue = article_node.find(".//JournalIssue")
            if journal_issue is not None:
                pub_date = journal_issue.find(".//PubDate")
                if pub_date is not None:
                    year = pub_date.findtext(".//Year")
                    if year:
                        pub_year = year

                # check Epublish date if no regular pub date
                if not pub_year:
                    pub_date_ep = journal_issue.find(".//PubDate[@PubStatus='epublish']")
                    if pub_date_ep is not None:
                        year = pub_date_ep.findtext(".//Year")
                        if year:
                            pub_year = year

            # Strategy B: Look in PubData
            if not pub_year:
                pub_data_node = article.find(".//PubData")
                if pub_data_node is not None:
                    article_date_node = pub_data_node.find(".//ArticleDate")
                    if article_date_node is not None:
                        year = article_date_node.findtext(".//Year")
                        if year:
                            pub_year = year

            # Strategy C: Fallback to "Electronic" or "Print" 
            if not pub_year:
                electronic_pub_date = article_node.find(".//ElectronicEpubDate")
                if electronic_pub_date is not None:
                    year = electronic_pub_date.findtext(".//Year")
                    if year:
                        pub_year = year
            
            pub_date_str = pub_year if pub_year else ""

            # -- 5. Extract Abstract
            abstract_node = article_node.find(".//Abstract")
            abstract_text = ""
            if abstract_node is not None:
                abstract_text = abstract_node.findtext(".//AbstractText")
                abstract_text = abstract_text.strip() if abstract_text else ""

            # Skip if essential data is missing
            if not title:
                continue
            
            # Combine
            papers.append({
                "title": title,
                "abstract": abstract_text,
                "authors": authors_str,
                "year": pub_date_str,
                "journal": journal_title,
                "pmid": article.findtext(".//PMID"),
                "source": "pubmed"
            })

        except Exception as e:
            print(f"Error processing PMID {article.findtext('.//PMID')}: {e}")
            continue

    return papers


# -----------------------------
# Step 3: Format for output
# -----------------------------
def format_papers_for_chat(papers: list, query_context: str = "") -> str:
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
            f"**{i}. {paper.get('title','Unknown Title')}**\n"
            f"\t • *Authors:* {paper.get('authors', 'unknown')}\n"
            f"\t • *Journal:* {paper.get('journal', 'unknown')} ({paper.get('year', 'unknown')})\n"
            f"\t • *PMID:* [{paper.get('pmid', 'N/A')}](https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid','')})\n"
            f"\t • *Summary:* {abstract_display}"
        )
        formatted_blocks.append(block)

    # create the header
    if query_context:
        header = f"Here are the top {len(papers)} papers for your query: *'{query_context}'*"
    else:
        header - f"Here are the details for {len(papers)} papers:"
        
    return f"{header}\n\n" + "\n---\n\n".join(formatted_blocks)


# -----------------------------
# Step 4: Full query pipeline
# -----------------------------
@tool
def build_abstracts_from_query(query="vision AND psychophysics", max_results=5):
    """
    Use this tool when the user asks you to search for papers on a topic.
    """
    pmids = search_pubmed(query, max_results)
    time.sleep(0.5)  # be polite to API
    papers = fetch_pubmed_details(pmids)
    return format_papers_for_chat(papers, query_context=query)



@tool
def search_arXiv(query: str) -> str:
    """
    Search arXiv and return raw paper titles and abstracts.
    """
    base_url = "http://export.arxiv.org/api/query"
    
    headers = {"User-Agent": "vision-research-bot/1.0"}
    
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 3
    }

    for attempt in range(3):
        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            break
        elif response.status_code == 429:
            time.sleep(2 ** attempt)
        else:
            return f"API error {response.status_code}"
    else:
        return "Failed after retries"

    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    output = []

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()

        title = " ".join(title.split())
        summary = " ".join(summary.split())

        output.append(
            f"Title: {title}\nSummary: {summary[:300]}...\n"
        )

    return "\n\n".join(output)