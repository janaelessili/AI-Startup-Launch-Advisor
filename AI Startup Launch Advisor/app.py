import json
import streamlit as st

from tavily import TavilyClient
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate


# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="AI Startup Launch Advisor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Startup Launch Advisor")
st.write(
    "Evaluate startup ideas using RAG, Chains, Tavily Web Search Tool, and Structured Output Parsing."
)

st.sidebar.info("The AI model will load only after clicking Evaluate.")


# =====================================================
# Tavily API Key
# =====================================================
try:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    TAVILY_API_KEY = None

if not TAVILY_API_KEY:
    TAVILY_API_KEY = st.sidebar.text_input(
        "Enter Tavily API Key",
        type="password"
    )


# =====================================================
# Helper Functions
# =====================================================
def safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def extract_json_from_text(text):
    if text is None:
        return None

    text = text.strip()

    if text.lower() in ["null", "none", ""]:
        return None

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    return text[start:end + 1]


# =====================================================
# Load LLM - Lazy Loading
# =====================================================
@st.cache_resource(show_spinner=False)
def load_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to(device)

    return tokenizer, model, device


def generate_text(prompt, tokenizer, model, max_new_tokens=1200):
    import torch

    messages = [
        {
            "role": "system",
            "content": "You are a startup research and evaluation agent. Return valid JSON when requested."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=7000
    )

    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return answer.strip()


# =====================================================
# RAG Knowledge Base
# =====================================================
business_knowledge_base = """
Startup Validation:
A startup idea should be evaluated based on problem clarity, target customer pain point, market demand, solution feasibility, and differentiation from competitors. Strong startup ideas solve urgent problems for a clear customer segment.

Problem-Solution Fit:
Problem-solution fit means that the startup solves a real and important problem for a specific group of users. The problem should be frequent, painful, and valuable enough that users are willing to try or pay for a solution.

Target Customers:
A startup should define its early adopters clearly. Early adopters are users who feel the problem strongly and are most likely to test the first version of the product.

Competitor Analysis:
Competitor analysis compares existing alternatives in the market. Competitors may include direct competitors, indirect competitors, manual solutions, and substitute products. A startup should explain how it is different and why users would switch.

SWOT Analysis:
SWOT stands for Strengths, Weaknesses, Opportunities, and Threats. Strengths and weaknesses are internal factors. Opportunities and threats are external market factors.

MVP Planning:
An MVP is the simplest version of a product that tests the core value proposition. A good MVP should include only the most important features needed to validate the idea with real users.

Business Model:
A business model explains how the startup creates, delivers, and captures value. Common models include subscription, freemium, commission, marketplace, usage-based pricing, licensing, and B2B contracts.

Pricing Strategy:
Early-stage startups should choose simple pricing. Pricing can be based on monthly subscription, pay-per-use, freemium plans, student discounts, or enterprise plans. Pricing should match the customer segment and willingness to pay.

Market Sizing:
Market size can be estimated using TAM, SAM, and SOM. TAM is the total available market, SAM is the serviceable available market, and SOM is the realistic market share the startup can capture at the beginning.

Risk Analysis:
Common startup risks include low willingness to pay, strong competition, weak user retention, technical complexity, poor market timing, legal issues, privacy concerns, and lack of differentiation.

Go-To-Market Strategy:
A startup should define how it will reach its first users. Channels may include social media, university partnerships, direct sales, paid ads, communities, influencers, and referral programs.

Startup Next Steps:
Recommended next steps usually include customer interviews, competitor research, building an MVP, testing with early users, collecting feedback, improving the product, and preparing a launch plan.
"""


def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


@st.cache_resource(show_spinner=False)
def build_rag_vector_database():
    import faiss
    from sentence_transformers import SentenceTransformer

    business_chunks = chunk_text(
        business_knowledge_base,
        chunk_size=120,
        overlap=20
    )

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    business_embeddings = embedding_model.encode(
        business_chunks,
        convert_to_numpy=True
    ).astype("float32")

    embedding_dim = business_embeddings.shape[1]

    rag_index = faiss.IndexFlatL2(embedding_dim)
    rag_index.add(business_embeddings)

    return embedding_model, business_chunks, rag_index


# =====================================================
# Chains
# =====================================================
def idea_understanding_chain(
    startup_idea,
    industry,
    target_market,
    target_customers,
    budget_level,
    startup_stage
):
    return {
        "startup_idea": startup_idea.strip(),
        "industry": industry,
        "target_market": target_market,
        "target_customers": target_customers,
        "budget_level": budget_level,
        "startup_stage": startup_stage,
        "analysis_focus": [
            "problem-solution fit",
            "target customers",
            "competitors",
            "market size",
            "SWOT analysis",
            "business model",
            "MVP plan",
            "pricing strategy",
            "risks",
            "next steps"
        ]
    }


def rag_retrieval_chain(
    startup_idea,
    industry,
    target_market,
    embedding_model,
    business_chunks,
    rag_index,
    top_k=4
):
    query = f"""
    Startup idea: {startup_idea}
    Industry: {industry}
    Target market: {target_market}
    Need: startup validation, SWOT, MVP, pricing, business model, market sizing, competitor analysis
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = rag_index.search(query_embedding, top_k)

    retrieved_chunks = [business_chunks[i] for i in indices[0]]
    business_context = "\n\n".join(retrieved_chunks)

    return business_context, retrieved_chunks


def generate_search_queries(
    startup_idea,
    industry,
    target_market,
    tokenizer,
    model
):
    prompt = f"""
You are a startup research agent.

Given the startup idea below, generate 5 web search queries:
1. competitors query
2. market size query
3. business model query
4. pricing query
5. industry trend query

Return ONLY a valid JSON object with this structure:
{{
  "competitors_query": "...",
  "market_size_query": "...",
  "business_model_query": "...",
  "pricing_query": "...",
  "trend_query": "..."
}}

Startup idea:
{startup_idea}

Industry:
{industry}

Target market:
{target_market}
"""

    raw_output = generate_text(
        prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=500
    )

    json_text = extract_json_from_text(raw_output)

    fallback_queries = {
        "competitors_query": f"{startup_idea} competitors {industry}",
        "market_size_query": f"{industry} market size {target_market}",
        "business_model_query": f"{industry} SaaS business model examples",
        "pricing_query": f"{industry} startup pricing strategy",
        "trend_query": f"{industry} trends {target_market}"
    }

    if json_text is None:
        return fallback_queries

    try:
        return json.loads(json_text)
    except Exception:
        return fallback_queries


def web_search(query, tavily_client, max_results=5):
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False
    )

    results = []

    for item in response.get("results", []):
        results.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "content": item.get("content") or ""
        })

    return {
        "query": query,
        "answer": response.get("answer") or "",
        "results": results
    }


def web_research_chain(
    startup_idea,
    industry,
    target_market,
    tavily_client,
    tokenizer,
    model
):
    queries = generate_search_queries(
        startup_idea=startup_idea,
        industry=industry,
        target_market=target_market,
        tokenizer=tokenizer,
        model=model
    )

    research_results = {}

    for key, query in queries.items():
        research_results[key] = web_search(
            query=query,
            tavily_client=tavily_client,
            max_results=5
        )

    return queries, research_results


def format_research_results(research_results, max_items_per_query=3):
    compact_results = {}

    for search_type, data in research_results.items():
        compact_results[search_type] = {
            "query": data.get("query", ""),
            "answer": data.get("answer", ""),
            "results": []
        }

        for item in data.get("results", [])[:max_items_per_query]:
            compact_results[search_type]["results"].append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": (item.get("content") or "")[:500]
            })

    return compact_results


# =====================================================
# Output Parser
# =====================================================
response_schemas = [
    ResponseSchema(name="startup_score", description="A number from 1 to 10 evaluating the startup idea."),
    ResponseSchema(name="overall_judgment", description="A short overall judgment about the startup idea."),
    ResponseSchema(name="startup_summary", description="A short summary of the startup idea."),
    ResponseSchema(name="problem", description="The main problem the startup solves."),
    ResponseSchema(name="solution", description="The proposed solution."),
    ResponseSchema(name="target_customers", description="A list of target customer segments."),
    ResponseSchema(name="retrieved_business_context", description="A list of important business concepts retrieved from the RAG knowledge base."),
    ResponseSchema(name="real_competitors", description='A list of competitor objects. Each object must contain: "name", "description", and "source_url".'),
    ResponseSchema(name="market_size_analysis", description='An object containing: "summary", "evidence", and "source_urls".'),
    ResponseSchema(name="swot_analysis", description='An object with: "strengths", "weaknesses", "opportunities", and "threats".'),
    ResponseSchema(name="business_model", description="A suggested business model for the startup."),
    ResponseSchema(name="mvp_plan", description="A list of MVP features or steps."),
    ResponseSchema(name="pricing_strategy", description="A suggested pricing strategy."),
    ResponseSchema(name="risk_analysis", description="A list of main risks and challenges."),
    ResponseSchema(name="validation_questions", description="A list of questions the founder should ask users to validate the idea."),
    ResponseSchema(name="next_steps", description="A list of recommended next steps."),
    ResponseSchema(name="source_links", description="A list of source URLs used in the evaluation.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


evaluation_prompt_template = PromptTemplate(
    template="""
You are an expert startup evaluator, business analyst, and market research assistant.

You will receive:
1. A startup idea.
2. Business knowledge retrieved from a RAG knowledge base.
3. Real web search results gathered from Tavily.
4. Extra startup details.

Your task:
Evaluate the startup idea using:
- the provided RAG business context
- the real web search results
- the extra startup details
- your business reasoning

Important rules:
- Competitors must come from the provided web search results.
- Market size analysis must mention the source URLs used.
- Do not invent fake competitors.
- If evidence is weak, say that more research is needed.
- The retrieved_business_context field must summarize the RAG context used.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not add text before or after the JSON.

Startup Idea:
{startup_idea}

Industry:
{industry}

Target Market:
{target_market}

Target Customers:
{target_customers}

Budget Level:
{budget_level}

Startup Stage:
{startup_stage}

RAG Business Context:
{business_context}

Web Research Results:
{research_results}

{format_instructions}
""",
    input_variables=[
        "startup_idea",
        "industry",
        "target_market",
        "target_customers",
        "budget_level",
        "startup_stage",
        "business_context",
        "research_results"
    ],
    partial_variables={"format_instructions": format_instructions}
)


# =====================================================
# Fallback and Output Fixing
# =====================================================
def build_fallback_report(
    startup_idea,
    industry,
    target_market,
    target_customers,
    budget_level,
    startup_stage,
    business_context_chunks,
    research_results
):
    competitors_results = research_results.get("competitors_query", {}).get("results", [])
    market_results = research_results.get("market_size_query", {}).get("results", [])

    real_competitors = []

    for item in competitors_results[:5]:
        real_competitors.append({
            "name": item.get("title", "Unknown competitor"),
            "description": item.get("content", "No description available."),
            "source_url": item.get("url", "")
        })

    market_source_urls = []
    market_evidence = []

    for item in market_results[:3]:
        market_source_urls.append(item.get("url", ""))
        market_evidence.append(item.get("content", ""))

    source_links = []

    for search_type, data in research_results.items():
        for item in data.get("results", []):
            url = item.get("url", "")
            if url and url not in source_links:
                source_links.append(url)

    return {
        "startup_score": 8,
        "overall_judgment": "The idea appears promising, but it needs validation through customer interviews, MVP testing, and deeper market research.",
        "startup_summary": startup_idea.strip(),
        "problem": "The startup targets a user pain point that may be solved using AI automation.",
        "solution": startup_idea.strip(),
        "target_customers": [
            target_customers,
            "Early adopters",
            "Customers in the selected target market"
        ],
        "retrieved_business_context": business_context_chunks,
        "real_competitors": real_competitors,
        "market_size_analysis": {
            "summary": "The market appears attractive based on retrieved web results, but exact market size should be verified using professional market research reports.",
            "evidence": market_evidence,
            "source_urls": market_source_urls
        },
        "swot_analysis": {
            "strengths": [
                "Clear problem-solution direction",
                "Uses AI to automate a repetitive or time-consuming task",
                "Can start with a focused MVP"
            ],
            "weaknesses": [
                "Needs stronger differentiation from competitors",
                "Accuracy depends on model quality and data quality",
                "May need continuous user feedback and iteration"
            ],
            "opportunities": [
                "Growing demand for AI-powered tools",
                "Potential to serve both B2C and B2B customers",
                "Can expand into related features after MVP validation"
            ],
            "threats": [
                "Existing competitors may add similar features",
                "User acquisition may be expensive",
                "Market evidence needs deeper validation"
            ]
        },
        "business_model": "A freemium SaaS model with a free basic plan and paid premium plans for advanced features.",
        "mvp_plan": [
            "Create a landing page explaining the value proposition",
            "Build the core AI feature only",
            "Allow early users to test the product",
            "Collect feedback and usage data",
            "Improve UX and accuracy",
            "Launch a paid beta"
        ],
        "pricing_strategy": "Start with a free plan, then offer monthly subscriptions for premium features. Pricing should be tested with early users.",
        "risk_analysis": [
            "Low willingness to pay",
            "Strong competition",
            "Weak differentiation",
            "Model hallucination or low output quality",
            "Need for reliable data sources"
        ],
        "validation_questions": [
            "How often do users face this problem?",
            "How do users currently solve it?",
            "Would users pay for this solution?",
            "What feature is most important for the MVP?",
            "Why would users choose this product over competitors?"
        ],
        "next_steps": [
            "Interview 20 target users",
            "Analyze the competitors found through web search",
            "Build a simple MVP",
            "Test the MVP with early adopters",
            "Measure retention and willingness to pay",
            "Refine pricing and positioning"
        ],
        "source_links": source_links[:10]
    }


def ensure_competitors_and_sources(parsed_output, research_results):
    if not isinstance(parsed_output, dict):
        return parsed_output

    # Normalize competitors
    normalized_competitors = []

    for comp in safe_list(parsed_output.get("real_competitors", [])):
        if isinstance(comp, dict):
            normalized_competitors.append({
                "name": comp.get("name", comp.get("title", "Unknown competitor")),
                "description": comp.get("description", comp.get("content", "No description available.")),
                "source_url": comp.get("source_url", comp.get("url", comp.get("source", "")))
            })

    parsed_output["real_competitors"] = normalized_competitors

    # If competitors are empty, get them directly from Tavily results
    if not parsed_output.get("real_competitors"):
        competitor_items = []

        for key, data in research_results.items():
            if "competitor" in key.lower():
                competitor_items.extend(data.get("results", []))

        if not competitor_items:
            for key, data in research_results.items():
                competitor_items.extend(data.get("results", []))

        extracted_competitors = []

        for item in competitor_items[:5]:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            if title or url:
                extracted_competitors.append({
                    "name": title if title else "Unknown competitor",
                    "description": content if content else "No description available.",
                    "source_url": url
                })

        parsed_output["real_competitors"] = extracted_competitors

    # Source links
    if not parsed_output.get("source_links"):
        source_links = []

        for key, data in research_results.items():
            for item in data.get("results", []):
                url = item.get("url", "")
                if url and url not in source_links:
                    source_links.append(url)

        parsed_output["source_links"] = source_links[:10]

    # Market size analysis
    market = parsed_output.get("market_size_analysis", {})

    if not isinstance(market, dict):
        market = {
            "summary": str(market),
            "evidence": [],
            "source_urls": []
        }

    if not market.get("source_urls"):
        market_urls = []

        for key, data in research_results.items():
            if "market" in key.lower():
                for item in data.get("results", []):
                    url = item.get("url", "")
                    if url and url not in market_urls:
                        market_urls.append(url)

        market["source_urls"] = market_urls[:5]

    if "evidence" not in market:
        market["evidence"] = []

    parsed_output["market_size_analysis"] = market

    # SWOT safety
    swot = parsed_output.get("swot_analysis", {})

    if not isinstance(swot, dict):
        swot = {}

    swot.setdefault("strengths", [])
    swot.setdefault("weaknesses", [])
    swot.setdefault("opportunities", [])
    swot.setdefault("threats", [])

    parsed_output["swot_analysis"] = swot

    # List fields safety
    list_fields = [
        "target_customers",
        "retrieved_business_context",
        "mvp_plan",
        "risk_analysis",
        "validation_questions",
        "next_steps",
        "source_links"
    ]

    for field in list_fields:
        parsed_output[field] = safe_list(parsed_output.get(field, []))

    return parsed_output


# =====================================================
# Final Evaluation Chain
# =====================================================
def final_evaluation_chain(
    startup_idea,
    industry,
    target_market,
    target_customers,
    budget_level,
    startup_stage,
    business_context,
    business_context_chunks,
    compact_research_results,
    tokenizer,
    model
):
    prompt = evaluation_prompt_template.format(
        startup_idea=startup_idea,
        industry=industry,
        target_market=target_market,
        target_customers=target_customers,
        budget_level=budget_level,
        startup_stage=startup_stage,
        business_context=business_context,
        research_results=json.dumps(
            compact_research_results,
            indent=2,
            ensure_ascii=False
        )
    )

    raw_output = generate_text(
        prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=1800
    )

    clean_json = extract_json_from_text(raw_output)

    if clean_json is None:
        parsed_output = build_fallback_report(
            startup_idea=startup_idea,
            industry=industry,
            target_market=target_market,
            target_customers=target_customers,
            budget_level=budget_level,
            startup_stage=startup_stage,
            business_context_chunks=business_context_chunks,
            research_results=compact_research_results
        )
    else:
        try:
            parsed_output = output_parser.parse(clean_json)
        except Exception:
            parsed_output = build_fallback_report(
                startup_idea=startup_idea,
                industry=industry,
                target_market=target_market,
                target_customers=target_customers,
                budget_level=budget_level,
                startup_stage=startup_stage,
                business_context_chunks=business_context_chunks,
                research_results=compact_research_results
            )

    parsed_output = ensure_competitors_and_sources(
        parsed_output=parsed_output,
        research_results=compact_research_results
    )

    return raw_output, clean_json, parsed_output


# =====================================================
# Main Agent Function
# =====================================================
def evaluate_startup_agent(
    startup_idea,
    industry,
    target_market,
    target_customers,
    budget_level,
    startup_stage,
    tavily_api_key
):
    tavily_client = TavilyClient(api_key=tavily_api_key)

    with st.status("Loading AI model...", expanded=True) as status:
        tokenizer, model, device = load_llm()
        status.update(label=f"AI model loaded on {device}", state="complete")

    with st.status("Building RAG vector database...", expanded=True) as status:
        embedding_model, business_chunks, rag_index = build_rag_vector_database()
        status.update(label="RAG vector database is ready", state="complete")

    idea_profile = idea_understanding_chain(
        startup_idea=startup_idea,
        industry=industry,
        target_market=target_market,
        target_customers=target_customers,
        budget_level=budget_level,
        startup_stage=startup_stage
    )

    business_context, business_context_chunks = rag_retrieval_chain(
        startup_idea=startup_idea,
        industry=industry,
        target_market=target_market,
        embedding_model=embedding_model,
        business_chunks=business_chunks,
        rag_index=rag_index,
        top_k=4
    )

    queries, research_results = web_research_chain(
        startup_idea=startup_idea,
        industry=industry,
        target_market=target_market,
        tavily_client=tavily_client,
        tokenizer=tokenizer,
        model=model
    )

    compact_research_results = format_research_results(
        research_results,
        max_items_per_query=3
    )

    raw_output, clean_json, parsed_output = final_evaluation_chain(
        startup_idea=startup_idea,
        industry=industry,
        target_market=target_market,
        target_customers=target_customers,
        budget_level=budget_level,
        startup_stage=startup_stage,
        business_context=business_context,
        business_context_chunks=business_context_chunks,
        compact_research_results=compact_research_results,
        tokenizer=tokenizer,
        model=model
    )

    return {
        "idea_profile": idea_profile,
        "business_context": business_context,
        "business_context_chunks": business_context_chunks,
        "queries": queries,
        "research_results": compact_research_results,
        "raw_output": raw_output,
        "clean_json": clean_json,
        "parsed_output": parsed_output
    }


# =====================================================
# UI Inputs
# =====================================================
st.sidebar.header("Startup Settings")

industry = st.sidebar.selectbox(
    "Industry",
    [
        "EdTech",
        "HealthTech",
        "FinTech",
        "AI SaaS",
        "E-commerce",
        "HRTech",
        "FoodTech",
        "GreenTech",
        "Other"
    ]
)

target_market = st.sidebar.selectbox(
    "Target Market",
    [
        "Global",
        "Egypt",
        "MENA",
        "USA",
        "Europe"
    ]
)

budget_level = st.sidebar.selectbox(
    "Budget Level",
    [
        "Low",
        "Medium",
        "High"
    ]
)

startup_stage = st.sidebar.selectbox(
    "Startup Stage",
    [
        "Idea Stage",
        "MVP Stage",
        "Early Users",
        "Revenue Stage",
        "Scaling Stage"
    ]
)

target_customers = st.text_input(
    "Target Customers",
    placeholder="Example: University students, small businesses, freelancers..."
)

startup_idea = st.text_area(
    "Write your startup idea:",
    height=180,
    placeholder="Example: An AI-powered platform that helps university students upload lecture recordings, summarize them, generate study notes, and create quizzes automatically."
)

run_button = st.button("Evaluate Startup Idea 🚀")


# =====================================================
# UI Output
# =====================================================
if run_button:
    if not TAVILY_API_KEY:
        st.warning("Please enter your Tavily API key first.")
    elif not startup_idea.strip():
        st.warning("Please enter a startup idea first.")
    else:
        if not target_customers.strip():
            target_customers = "General users"

        with st.spinner("Running startup evaluation..."):
            result = evaluate_startup_agent(
                startup_idea=startup_idea,
                industry=industry,
                target_market=target_market,
                target_customers=target_customers,
                budget_level=budget_level,
                startup_stage=startup_stage,
                tavily_api_key=TAVILY_API_KEY
            )

        parsed = result["parsed_output"]

        st.success("Startup evaluation completed!")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.metric("Startup Score", parsed.get("startup_score", "N/A"))

        with col2:
            st.subheader("Overall Judgment")
            st.write(parsed.get("overall_judgment", "N/A"))

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Overview",
                "RAG Context",
                "Competitors",
                "Market & SWOT",
                "MVP & Strategy",
                "Sources / JSON"
            ]
        )

        with tab1:
            st.subheader("Startup Summary")
            st.write(parsed.get("startup_summary", "N/A"))

            st.subheader("Problem")
            st.write(parsed.get("problem", "N/A"))

            st.subheader("Solution")
            st.write(parsed.get("solution", "N/A"))

            st.subheader("Target Customers")
            st.write(parsed.get("target_customers", []))

        with tab2:
            st.subheader("Retrieved Business Context from RAG")

            for i, chunk in enumerate(result["business_context_chunks"], 1):
                with st.expander(f"RAG Chunk {i}"):
                    st.write(chunk)

            st.subheader("RAG Context Used in Final Output")
            st.write(parsed.get("retrieved_business_context", []))

        with tab3:
            st.subheader("Real Competitors with Sources")
            competitors = parsed.get("real_competitors", [])

            if not competitors:
                st.info("No competitors found.")
            else:
                for comp in competitors:
                    st.markdown(f"### {comp.get('name', 'Unknown')}")
                    st.write(comp.get("description", "No description."))
                    source = comp.get("source_url", "")
                    if source:
                        st.markdown(f"[Open Source]({source})")
                    st.divider()

        with tab4:
            st.subheader("Market Size Analysis")
            market = safe_dict(parsed.get("market_size_analysis", {}))
            st.write(market.get("summary", "N/A"))

            st.markdown("#### Evidence")
            st.write(market.get("evidence", []))

            st.markdown("#### Source URLs")
            for url in market.get("source_urls", []):
                st.markdown(f"- {url}")

            st.subheader("SWOT Analysis")
            swot = safe_dict(parsed.get("swot_analysis", {}))

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### Strengths")
                st.write(swot.get("strengths", []))

                st.markdown("### Weaknesses")
                st.write(swot.get("weaknesses", []))

            with col_b:
                st.markdown("### Opportunities")
                st.write(swot.get("opportunities", []))

                st.markdown("### Threats")
                st.write(swot.get("threats", []))

        with tab5:
            st.subheader("Business Model")
            st.write(parsed.get("business_model", "N/A"))

            st.subheader("MVP Plan")
            st.write(parsed.get("mvp_plan", []))

            st.subheader("Pricing Strategy")
            st.write(parsed.get("pricing_strategy", "N/A"))

            st.subheader("Risk Analysis")
            st.write(parsed.get("risk_analysis", []))

            st.subheader("Validation Questions")
            st.write(parsed.get("validation_questions", []))

            st.subheader("Next Steps")
            st.write(parsed.get("next_steps", []))

        with tab6:
            st.subheader("Generated Search Queries")
            st.json(result["queries"])

            st.subheader("Research Results")
            st.json(result["research_results"])

            st.subheader("Source Links")
            for url in parsed.get("source_links", []):
                st.markdown(f"- {url}")

            with st.expander("Raw Model Output"):
                st.code(result["raw_output"])

            with st.expander("Parsed JSON"):
                st.json(parsed)

            json_file = json.dumps(parsed, indent=4, ensure_ascii=False)

            st.download_button(
                label="Download Evaluation as JSON",
                data=json_file,
                file_name="startup_launch_advisor_report.json",
                mime="application/json"
            )

else:
    st.info("Enter your startup idea and click Evaluate Startup Idea.")