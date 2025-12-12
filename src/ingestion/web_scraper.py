"""Web scraper for automatic document discovery and ingestion."""

import hashlib
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ..utils import get_logger, get_settings
from .pdf_to_markdown import convert_pdf_to_markdown

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ScrapedDocument:
    """Scraped document information."""

    url: str
    title: str
    content_type: str
    file_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MicrobiomeScraper:
    """
    Web scraper for microbiome research documents.
    
    Targets common microbiome research repositories and journals:
    - PubMed Central
    - bioRxiv
    - medRxiv
    - NIH Human Microbiome Project
    - Microbiome journal
    """
    
    # Common microbiome-related keywords
    KEYWORDS = [
        "microbiome",
        "microbiota",
        "gut bacteria",
        "16S rRNA",
        "metagenomics",
        "metatranscriptomics",
        "metabolomics",
        "dysbiosis",
        "probiotic",
        "prebiotic",
    ]
    
    # Target domains for PDF discovery
    PDF_DOMAINS = [
        "ncbi.nlm.nih.gov",
        "biorxiv.org",
        "medrxiv.org",
        "microbiomejournal.biomedcentral.com",
        "nature.com",
        "cell.com",
        "pnas.org",
    ]
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        download_dir: Optional[Path] = None,
        max_pages: int = 10,
        delay: float = 1.0,
    ):
        """
        Initialize scraper.
        
        Args:
            output_dir: Directory for processed markdown
            download_dir: Directory for downloaded PDFs
            max_pages: Maximum pages to scrape per source
            delay: Delay between requests (seconds)
        """
        self.output_dir = output_dir or settings.ingestion.markdown_dir
        self.download_dir = download_dir or Path(tempfile.mkdtemp())
        self.max_pages = max_pages
        self.delay = delay
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KnightGPT Research Bot/1.0 (Academic research)",
        })
        
        # Track processed URLs to avoid duplicates
        self._processed_urls: set[str] = set()

    def scrape_pubmed(
        self,
        query: str = "microbiome",
        max_results: int = 100,
    ) -> list[ScrapedDocument]:
        """
        Scrape PubMed Central for open-access PDFs.
        
        Args:
            query: Search query
            max_results: Maximum results to fetch
            
        Returns:
            List of scraped documents
        """
        logger.info(f"Scraping PubMed for: {query}")
        
        # PubMed E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Search for articles
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {
            "db": "pmc",
            "term": f"{query}[Title/Abstract] AND open access[filter]",
            "retmax": max_results,
            "retmode": "json",
        }
        
        try:
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            data = response.json()
            
            ids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(ids)} PubMed Central articles")
            
            documents = []
            for pmc_id in ids[:max_results]:
                doc = self._fetch_pmc_article(pmc_id)
                if doc:
                    documents.append(doc)
                    
            return documents
            
        except Exception as e:
            logger.error(f"PubMed scraping failed: {e}")
            return []

    def _fetch_pmc_article(self, pmc_id: str) -> Optional[ScrapedDocument]:
        """Fetch a single PMC article."""
        # Check for PDF link
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
        
        if pdf_url in self._processed_urls:
            return None
            
        try:
            response = self.session.head(pdf_url, allow_redirects=True)
            
            if response.status_code == 200:
                self._processed_urls.add(pdf_url)
                return self._download_and_process(pdf_url, f"PMC{pmc_id}")
                
        except Exception as e:
            logger.debug(f"Failed to fetch PMC{pmc_id}: {e}")
            
        return None

    def scrape_biorxiv(
        self,
        query: str = "microbiome",
        max_results: int = 100,
    ) -> list[ScrapedDocument]:
        """
        Scrape bioRxiv for preprints.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of scraped documents
        """
        logger.info(f"Scraping bioRxiv for: {query}")
        
        # bioRxiv API
        api_url = "https://api.biorxiv.org/details/biorxiv"
        
        # Get recent papers (last 30 days)
        from datetime import timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        try:
            url = f"{api_url}/{start_date}/{end_date}"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            documents = []
            for article in data.get("collection", [])[:max_results]:
                # Filter by keywords
                title = article.get("title", "").lower()
                abstract = article.get("abstract", "").lower()
                
                if any(kw in title or kw in abstract for kw in self.KEYWORDS):
                    doi = article.get("doi")
                    if doi:
                        pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
                        doc = self._download_and_process(pdf_url, doi.replace("/", "_"))
                        if doc:
                            doc.metadata["title"] = article.get("title")
                            doc.metadata["doi"] = doi
                            documents.append(doc)
                            
            return documents
            
        except Exception as e:
            logger.error(f"bioRxiv scraping failed: {e}")
            return []

    def scrape_url(self, url: str) -> list[ScrapedDocument]:
        """
        Scrape a single URL for PDF links.
        
        Args:
            url: URL to scrape
            
        Returns:
            List of discovered documents
        """
        logger.info(f"Scraping URL: {url}")
        
        if url in self._processed_urls:
            return []
            
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            
            # Direct PDF
            if "pdf" in content_type.lower():
                self._processed_urls.add(url)
                doc = self._download_and_process(url, self._url_to_filename(url))
                return [doc] if doc else []
            
            # HTML page - find PDF links
            soup = BeautifulSoup(response.text, "html.parser")
            pdf_links = self._find_pdf_links(soup, url)
            
            documents = []
            for pdf_url in pdf_links:
                if pdf_url not in self._processed_urls:
                    self._processed_urls.add(pdf_url)
                    doc = self._download_and_process(
                        pdf_url, 
                        self._url_to_filename(pdf_url)
                    )
                    if doc:
                        documents.append(doc)
                        
            return documents
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return []

    def _find_pdf_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Find PDF links in HTML."""
        pdf_links = []
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            
            # Check if link points to PDF
            if href.lower().endswith(".pdf") or "/pdf/" in href.lower():
                full_url = urljoin(base_url, href)
                pdf_links.append(full_url)
                
        return pdf_links

    def _download_and_process(
        self,
        url: str,
        name: str,
    ) -> Optional[ScrapedDocument]:
        """Download PDF and convert to markdown."""
        logger.debug(f"Downloading: {url}")
        
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Verify it's a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                return None
            
            # Save PDF
            pdf_path = self.download_dir / f"{name}.pdf"
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert to markdown
            result = convert_pdf_to_markdown(
                pdf_path=pdf_path,
                output_dir=self.output_dir,
            )
            
            if result.get("success"):
                return ScrapedDocument(
                    url=url,
                    title=name,
                    content_type="application/pdf",
                    file_path=Path(result["output_path"]),
                    metadata=result.get("metadata", {}),
                )
                
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            
        return None

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename."""
        parsed = urlparse(url)
        path = parsed.path.replace("/", "_")
        
        # Create hash for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Clean filename
        name = re.sub(r"[^\w\-_]", "", path)[:50]
        return f"{name}_{url_hash}"


def run_scheduled_scrape(
    sources: list[str] = None,
    keywords: list[str] = None,
    output_dir: Optional[Path] = None,
) -> list[ScrapedDocument]:
    """
    Run scheduled scraping job.
    
    Args:
        sources: List of source URLs
        keywords: Search keywords
        output_dir: Output directory
        
    Returns:
        List of scraped documents
    """
    scraper = MicrobiomeScraper(output_dir=output_dir)
    
    keywords = keywords or MicrobiomeScraper.KEYWORDS[:3]
    documents = []
    
    # Scrape PubMed
    for keyword in keywords:
        docs = scraper.scrape_pubmed(query=keyword, max_results=20)
        documents.extend(docs)
    
    # Scrape bioRxiv
    for keyword in keywords:
        docs = scraper.scrape_biorxiv(query=keyword, max_results=20)
        documents.extend(docs)
    
    # Scrape custom URLs
    if sources:
        for url in sources:
            docs = scraper.scrape_url(url)
            documents.extend(docs)
    
    logger.info(f"Scraped {len(documents)} documents total")
    return documents


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape microbiome documents")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/markdown"))
    parser.add_argument("--keywords", "-k", nargs="+", default=["microbiome"])
    parser.add_argument("--url", "-u", type=str, help="Single URL to scrape")
    
    args = parser.parse_args()
    
    scraper = MicrobiomeScraper(output_dir=args.output)
    
    if args.url:
        docs = scraper.scrape_url(args.url)
    else:
        docs = run_scheduled_scrape(
            keywords=args.keywords,
            output_dir=args.output,
        )
    
    print(f"Scraped {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.title}: {doc.file_path}")
