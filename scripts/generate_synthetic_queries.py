import json
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.evaluation import generate_qa_embedding_pairs
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core.node_parser import SentenceSplitter
import tiktoken
load_dotenv(Path(__file__).parent.parent / ".env", override=True)


NORWEGIAN_QA_PROMPT_GENERAL = """\
Kontekst informasjon er nedenfor.

---------------------
{context_str}
---------------------

Gitt kontekst informasjonen og ingen forhåndskunnskap, generer {num_questions_per_chunk} spørsmål basert på konteksten.

Spørsmålene skal:
- Være på norsk
- Være spørsmål som kan besvares av informasjonen i konteksten
- Være varierte og dekke ulike aspekter av konteksten
- Være konkrete og spesifikke
- Være spørsmål en vanlig bruker ville stilt

Generer kun spørsmålene, ett per linje, uten nummerering eller annen formatering.
"""
NORWEGIAN_QA_PROMPT = """\
Kontekst informasjon er nedenfor.

---------------------
{context_str}
---------------------

Gitt kontekst informasjonen og ingen forhåndskunnskap, generer {num_questions_per_chunk} spørsmål basert på konteksten.

Spørsmålene skal:
- Være på norsk
- Være spørsmål som kan besvares av informasjonen i konteksten
- Være varierte og dekke ulike aspekter av konteksten
- Være konkrete og spesifikke
- Være spørsmål en vanlig bruker ville stilt
- Bruker er en forelder med barn med sammensatte helseutfordringer med behov for koordinerte tjenester; fokuser på spørsmål som er relevante for denne målgruppen
- Bruker er ikke ekspert, unngå tekniske spørsmål og fokusér på praktiske og forståelige spørsmål en vanlig bruker ville stilt

Generer kun spørsmålene, ett per linje, uten nummerering eller annen formatering.
Eksempel på spørsmål en bruker kunne stilt basert på konteksten:
- Hva slags økonomisk støtte finnes for familier med barn med nedsatt funksjon og stort omsorgsbehov?
- Hva er individuell plan og ansvarsgruppe?
- Hvordan kan vi tilpasse boligen til barnets behov?
- Hva er Brukerstyrt personlig assistanse (BPA)?


"""

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Tell antall tokens i en tekst.
    
    Args:
        text: Teksten å telle tokens for
        model: Modell å bruke for tokenisering
    
    Returns:
        Antall tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def load_documents_from_jsonl(
    file_path: Path, 
    max_docs: int = None,
    min_tokens: int = None,
    max_tokens: int = None
) -> list[Document]:
    """
    Last inn dokumenter fra JSONL fil og konverter til LlamaIndex Documents.
    
    Args:
        file_path: Sti til JSONL fil
        max_docs: Maksimalt antall dokumenter å laste (None = alle)
        min_tokens: Minimum antall tokens per dokument (None = ingen filter)
        max_tokens: Maksimum antall tokens per dokument (None = ingen filter)
    
    Returns:
        Liste med LlamaIndex Document objekter
    """
    documents = []
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_empty = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and len(documents) >= max_docs:
                break
                
            data = json.loads(line)
            
            # Kombinér 'tittel' og 'innhold' - begge inneholder tekst
            tittel = data.get('tittel', '')
            innhold = data.get('innhold', '')
            content = data.get('content', '')  # fallback for andre datasett
            
            # Bruk innhold hvis det finnes, ellers tittel
            # (de ser ut til å være like basert på eksempelet)
            text = innhold if innhold and innhold.strip() else tittel
            if not text and content:
                text = content
            # Skip tomme dokumenter
            if not text or not text.strip():
                skipped_empty += 1
                continue
            
            # Tell tokens
            token_count = count_tokens(text)
            
            # Filtrer basert på token lengde
            if min_tokens and token_count < min_tokens:
                skipped_too_short += 1
                continue
            
            if max_tokens and token_count > max_tokens:
                skipped_too_long += 1
                continue
            
            # Minimal metadata - kun ID og URL
            doc = Document(
                text=text,
                metadata={
                    'doc_id': data.get('dokument_id', ''),
                    'url': data.get('url', ''),
                }
            )
            
            documents.append(doc)
    
    # Statistikk
    print(f"   📊 Statistikk:")
    print(f"      ✓ Akseptert: {len(documents)} dokumenter")
    if skipped_empty > 0:
        print(f"      ⊘ Tomme: {skipped_empty}")
    if min_tokens and skipped_too_short > 0:
        print(f"      ⊘ For korte (< {min_tokens} tokens): {skipped_too_short}")
    if max_tokens and skipped_too_long > 0:
        print(f"      ⊘ For lange (> {max_tokens} tokens): {skipped_too_long}")
    
    if documents:
        token_counts = [count_tokens(doc.text) for doc in documents]
        print(f"      📈 Token range: {min(token_counts)} - {max(token_counts)}")
        print(f"      📊 Gjennomsnitt: {sum(token_counts) // len(token_counts)} tokens")
    
    return documents
    
def analyze_dataset(file_path: Path, sample_size: int = None):
    """
    Analyser et datasett for å finne token-distribusjon og kategori-fordeling.
    
    Args:
        file_path: Sti til JSONL fil
        sample_size: Antall dokumenter å analysere (None = alle)
    """
    print(f"\n📊 ANALYSERER: {file_path}")
    print("=" * 60)
    
    token_counts = []
    categories_nivaa1 = {}
    categories_nivaa2 = {}
    categories_nivaa3 = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            
            data = json.loads(line)
            
            # Bruk innhold som primær kilde
            text = data.get('innhold', '') or data.get('tittel', '') or data.get('content')
            
            if text:
                tokens = count_tokens(text)
                token_counts.append(tokens)
            
            # Tell kategorier
            nivaa1 = data.get('nivaa1', 'Ukjent')
            nivaa2 = data.get('nivaa2', 'Ukjent')
            nivaa3 = data.get('nivaa3', 'Ukjent')
            
            # Håndter NaN verdier
            if nivaa1 != nivaa1 or nivaa1 is None:  # NaN check
                nivaa1 = 'Ukjent'
            if nivaa2 != nivaa2 or nivaa2 is None:
                nivaa2 = 'Ukjent'
            if nivaa3 != nivaa3 or nivaa3 is None:
                nivaa3 = 'Ukjent'
            
            categories_nivaa1[nivaa1] = categories_nivaa1.get(nivaa1, 0) + 1
            categories_nivaa2[nivaa2] = categories_nivaa2.get(nivaa2, 0) + 1
            categories_nivaa3[nivaa3] = categories_nivaa3.get(nivaa3, 0) + 1
    
    # ==========================================
    # TOKEN STATISTIKK
    # ==========================================
    if token_counts:
        token_counts.sort()
        
        print(f"\n📏 TOKEN STATISTIKK:")
        print(f"Analyserte {len(token_counts)} dokumenter:")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Gjennomsnitt: {sum(token_counts) // len(token_counts)}")
        print(f"  Median: {token_counts[len(token_counts)//2]}")
        print(f"\nPercentiler:")
        print(f"  10%: {token_counts[len(token_counts)//10]}")
        print(f"  25%: {token_counts[len(token_counts)//4]}")
        print(f"  50%: {token_counts[len(token_counts)//2]}")
        print(f"  75%: {token_counts[3*len(token_counts)//4]}")
        print(f"  90%: {token_counts[9*len(token_counts)//10]}")
        
        # Foreslå chunk size
        suggested_chunk = token_counts[3*len(token_counts)//4] + 500
        print(f"\n💡 Foreslått chunk size: {suggested_chunk}")
    
    # ==========================================
    # KATEGORI STATISTIKK
    # ==========================================
    total_docs = sum(categories_nivaa1.values())
    
    # Nivå 1 - Hovedkategorier
    if categories_nivaa1:
        print(f"\n📂 NIVÅ 1 - HOVEDKATEGORIER:")
        print(f"Totalt {len(categories_nivaa1)} unike kategorier\n")
        
        sorted_cat1 = sorted(categories_nivaa1.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_cat1:
            percentage = (count / total_docs) * 100
            bar_length = int(percentage / 2)  # Scale for visning (max 50 tegn)
            bar = "█" * bar_length
            print(f"  {category:45s} {count:5d} docs ({percentage:5.1f}%) {bar}")
        
        print(f"\n  Total: {total_docs} dokumenter")
    
    # Nivå 2 - Underkategorier (kun vis hvis ikke for mange)
    if categories_nivaa2 and len(categories_nivaa2) < 50:
        print(f"\n📂 NIVÅ 2 - UNDERKATEGORIER:")
        print(f"Totalt {len(categories_nivaa2)} unike kategorier\n")
        
        sorted_cat2 = sorted(categories_nivaa2.items(), key=lambda x: x[1], reverse=True)
        
        # Vis topp 20
        for category, count in sorted_cat2[:20]:
            percentage = (count / total_docs) * 100
            bar_length = int(percentage / 2)
            bar = "█" * min(bar_length, 40)  # Max 40 tegn
            print(f"  {category:45s} {count:5d} ({percentage:5.1f}%) {bar}")
        
        if len(sorted_cat2) > 20:
            print(f"\n  ... og {len(sorted_cat2) - 20} flere kategorier")
    
    elif categories_nivaa2 and len(categories_nivaa2) >= 50:
        print(f"\n📂 NIVÅ 2: {len(categories_nivaa2)} unike kategorier (for mange til å vise)")
    
    # Nivå 3 - kun antall
    if categories_nivaa3:
        unique_nivaa3 = len([k for k in categories_nivaa3.keys() if k != 'Ukjent'])
        print(f"\n📂 NIVÅ 3: {unique_nivaa3} unike kategorier")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Tell antall tokens i en tekst.
    
    Args:
        text: Teksten å telle tokens for
        model: Modell å bruke for tokenisering
    
    Returns:
        Antall tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def main():
    """Hovedfunksjon for å generere syntetiske QA-par."""


    # TESTING KONFIGURASJON - sett til None for å bruke alle
    TESTING_MODE = os.getenv("TESTING_MODE", "true").lower() == "true"
    MAX_TRAIN_DOCS = int(os.getenv("MAX_TRAIN_DOCS", "10")) if TESTING_MODE else None
    MAX_TEST_DOCS = int(os.getenv("MAX_TEST_DOCS", "10")) if TESTING_MODE else None

    MIN_TOKENS = int(os.getenv("MIN_TOKENS", "2000"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))

    # Konfigurasjon
    DATA_DIR = Path("data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # dataset_train.jsonl og dataset_test.jsonl skal inneholde dokumentene som skal brukes for henholdsvis train og test. Disse må legges i data/raw/ før kjøring.
    #TRAIN_FILE = RAW_DIR / "dataset_train.jsonl"
    #TEST_FILE = RAW_DIR / "dataset_test.jsonl"
    #TRAIN_OUTPUT = PROCESSED_DIR / "train_dataset_adv.json"
    #TEST_OUTPUT = PROCESSED_DIR / "test_dataset_adv.json"

    # eti_train.jsonl og eti_test.jsonl skal inneholde dokumentene som skal brukes for henholdsvis train og test. Disse må legges i data/raw/ før kjøring.
    TRAIN_FILE = RAW_DIR / "eti_train.jsonl"
    TEST_FILE = RAW_DIR / "eti_test.jsonl"
    TRAIN_OUTPUT = PROCESSED_DIR / "eti_train_smpl.json"
    TEST_OUTPUT = PROCESSED_DIR / "eti_test_smpl.json"
    
    # LlamaIndex konfigurasjon
    NUM_QUESTIONS_PER_CHUNK = 20  # Antall spørsmål per chunk
    
    # Chunking konfigurasjon (valgfritt)

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    
    # Azure OpenAI konfigurasjon fra .env
    if os.getenv("LLM_PROVIDER", "openai").lower() == "openai":
        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        AZURE_DEPLOYMENT_NAME_MINI = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_MINI", "gpt-4o-mini")

        if not AZURE_API_KEY:
            print("❌ AZURE_API_KEY ikke satt")
            return
    elif os.getenv("LLM_PROVIDER", "anthropic").lower() == "anthropic":
        AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY")
        AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
        AZURE_AI_DEPLOYMENT_NAME = os.getenv("AZURE_AI_DEPLOYMENT_NAME")
        if not AZURE_AI_API_KEY:
            print("❌ AZURE_AI_API_KEY ikke satt")
            return
    else:
        print("❌ Ugyldig LLM_PROVIDER i .env. Sett til 'openai' eller 'anthropic'.")
        return
    
    # Analyse mode - sett til true for å kun analysere data
    ANALYZE_ONLY = os.getenv("ANALYZE_ONLY", "false").lower() == "true"
    
    if TESTING_MODE:
        print("\n" + "🧪" * 30)
        print("🧪 TESTING MODE AKTIVERT")
        print(f"🧪 Bruker max {MAX_TRAIN_DOCS} train docs og {MAX_TEST_DOCS} test docs")
        print("🧪" * 30 + "\n")


    
    # Sjekk at input-filer eksisterer
    if not TRAIN_FILE.exists():
        print(f"❌ Feil: {TRAIN_FILE} finnes ikke")
        print(f"💡 Legg train.jsonl i {RAW_DIR}/")
        return
    
    if not TEST_FILE.exists():
        print(f"❌ Feil: {TEST_FILE} finnes ikke")
        print(f"💡 Legg test.jsonl i {RAW_DIR}/")
        return
    
    # Hvis ANALYZE_ONLY, kun kjør analyse
    if ANALYZE_ONLY:
        analyze_dataset(TRAIN_FILE, sample_size=None)
        analyze_dataset(TEST_FILE, sample_size=None)
        return
    
        # Sjekk API key
    
    
    
     # Initialiser Azure OpenAI LLM
    print(f"\n🔧 Konfigurasjon:")

    print(f"   Chunk size: {CHUNK_SIZE} tokens")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"   Token filter: {MIN_TOKENS} - {MAX_TOKENS} tokens")
    print(f"   Questions per chunk: {NUM_QUESTIONS_PER_CHUNK}")

    # Initialiser LLM
    if os.getenv("LLM_PROVIDER", "openai").lower() == "openai":
        print(f"   Azure Endpoint: {AZURE_ENDPOINT}")
        print(f"   Deployment: {AZURE_DEPLOYMENT_NAME}")
        print(f"   API Version: {AZURE_API_VERSION}")
        llm_mini = AzureOpenAI(
            model=AZURE_DEPLOYMENT_NAME_MINI,
            deployment_name=AZURE_DEPLOYMENT_NAME_MINI,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            temperature=0.7,
        )
        llm = AzureOpenAI(
            model=AZURE_DEPLOYMENT_NAME,
            deployment_name=AZURE_DEPLOYMENT_NAME,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            temperature=0.7,
        )

    elif os.getenv("LLM_PROVIDER", "anthropic").lower() == "anthropic":
        print(f"   Azure Endpoint: {AZURE_AI_ENDPOINT}")
        print(f"   Deployment: {AZURE_AI_DEPLOYMENT_NAME}")
        # Initialiser LLM med Anthropic via Azure AI Foundry
        from langchain_anthropic import ChatAnthropic 
        from llama_index.llms.langchain import LangChainLLM
        langchain_llm = ChatAnthropic(
            model=AZURE_AI_DEPLOYMENT_NAME,
            api_key=AZURE_AI_API_KEY,
            base_url=AZURE_AI_ENDPOINT,
            default_headers={"api-key": AZURE_AI_API_KEY},
            max_tokens=4096,
            temperature=0.7,
        )
        #kun 1 modell så bruker samme for mini og full
        llm_mini = LangChainLLM(llm=langchain_llm)
        llm = LangChainLLM(llm=langchain_llm)
    else:
        print("❌ Ugyldig LLM_PROVIDER i .env. Sett til 'openai' eller 'anthropic'.")
        return

    # Node parser for chunking (hvis dokumentene er lange)
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # ==========================================
    # PROSESSER TRAIN SET
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 PROSESSERER TRAIN SET")
    print("=" * 60)
    
    print(f"Laster dokumenter fra {TRAIN_FILE}...")
    train_documents = load_documents_from_jsonl(
        TRAIN_FILE, 
        max_docs=MAX_TRAIN_DOCS,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS
    )
    print(f"✓ Lastet {len(train_documents)} dokumenter")

    if len(train_documents) == 0:
        print("❌ Ingen dokumenter ble lastet. Sjekk token-filtrene.")
        return
    
    
    print(f"Lager nodes (chunking med {CHUNK_SIZE} tokens)...")
    train_nodes = node_parser.get_nodes_from_documents(train_documents)
    print(f"✓ Laget {len(train_nodes)} nodes")
    
    print(f"Genererer QA-par ({NUM_QUESTIONS_PER_CHUNK} spørsmål per chunk)...")

    train_dataset = generate_qa_embedding_pairs(
        llm=llm_mini,
        nodes=train_nodes,
        num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK,
        qa_generate_prompt_tmpl=NORWEGIAN_QA_PROMPT,
    )
    train_dataset.save_json(str(TRAIN_OUTPUT))
    print(f"✓ Genererte {len(train_dataset.queries)} train queries")
    print(f"✓ Lagret til {TRAIN_OUTPUT}")
    
    # ==========================================
    # PROSESSER TEST SET
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 PROSESSERER TEST SET")
    print("=" * 60)
    
    print(f"Laster dokumenter fra {TEST_FILE}...")
    test_documents = load_documents_from_jsonl(
        TEST_FILE, 
        max_docs=MAX_TEST_DOCS,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS
    )
    print(f"✓ Lastet {len(test_documents)} dokumenter")
    if len(test_documents) == 0:
        print("❌ Ingen dokumenter ble lastet. Sjekk token-filtrene.")
        return
    print(f"Lager nodes (chunking med {CHUNK_SIZE} tokens)...")
    test_nodes = node_parser.get_nodes_from_documents(test_documents)
    print(f"✓ Laget {len(test_nodes)} nodes")
    
    print(f"Genererer QA-par ({NUM_QUESTIONS_PER_CHUNK} spørsmål per chunk)...")

    test_dataset = generate_qa_embedding_pairs(
        llm=llm,
        nodes=test_nodes,
        num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK,
        qa_generate_prompt_tmpl=NORWEGIAN_QA_PROMPT,
    )
    test_dataset.save_json(str(TEST_OUTPUT))
    print(f"✓ Genererte {len(test_dataset.queries)} test queries")
    print(f"✓ Lagret til {TEST_OUTPUT}")
    
    # ==========================================
    # OPPSUMMERING
    # ==========================================
    print("\n" + "=" * 60)
    print("✅ FERDIG!")
    print("=" * 60)
    print(f"Train:")
    print(f"  - Dokumenter: {len(train_documents)}")
    print(f"  - Nodes: {len(train_nodes)}")
    print(f"  - Queries: {len(train_dataset.queries)}")
    print(f"\nTest:")
    print(f"  - Dokumenter: {len(test_documents)}")
    print(f"  - Nodes: {len(test_nodes)}")
    print(f"  - Queries: {len(test_dataset.queries)}")
    print(f"\nTotal queries: {len(train_dataset.queries) + len(test_dataset.queries)}")
    print(f"\n📁 Filer lagret i: {PROCESSED_DIR}/")
    
    # Vis eksempel på genererte queries
    print("\n" + "=" * 60)
    print("📝 EKSEMPEL PÅ GENERERTE QUERIES")
    print("=" * 60)
    for i, (query_id, query) in enumerate(list(train_dataset.queries.items())[:3]):
        print(f"\nQuery {i+1}:")
        print(f"  ID: {query_id}")
        print(f"  Spørsmål: {query}")
        if query_id in train_dataset.relevant_docs:
            print(f"  Relevante docs: {len(train_dataset.relevant_docs[query_id])}")


if __name__ == "__main__":
    main()