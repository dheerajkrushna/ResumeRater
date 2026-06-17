"""
config.py — Central configuration for the Intelligent Candidate Ranking System.

All scoring weights, skill categories, company classifications, and thresholds
are defined here to allow easy tuning without modifying scoring logic.
"""

from datetime import datetime

# ============================================================================
# Reference date for the dataset (competition context: June 2026)
# ============================================================================
REFERENCE_DATE = datetime(2026, 6, 17)

# ============================================================================
# Scoring Component Weights (sum to 1.0)
# ============================================================================
WEIGHT_SEMANTIC = 0.30       # TF-IDF cosine similarity with JD
WEIGHT_RULE_BASED = 0.50     # Rule-based feature scoring
WEIGHT_BEHAVIORAL = 0.20     # Behavioral / platform engagement signals

# Rule-based sub-component weights (within the 0.50 rule-based allocation)
# These are raw point budgets that get normalized
BUDGET_TITLE = 15.0
BUDGET_EXPERIENCE = 15.0
BUDGET_COMPANY = 20.0
BUDGET_SKILLS = 35.0
BUDGET_TOTAL = BUDGET_TITLE + BUDGET_EXPERIENCE + BUDGET_COMPANY + BUDGET_SKILLS

# ============================================================================
# Job Description Text (embedded for TF-IDF vectorization)
# ============================================================================
JD_TEXT = """
Senior AI Engineer founding team. Own the intelligence layer: ranking, retrieval, 
matching systems for recruiters and candidates. Ship v2 ranking system with embeddings, 
hybrid retrieval, LLM-based re-ranking. Build evaluation infrastructure with offline 
benchmarks, online A/B testing, recruiter feedback loops.

Required: Production experience with embeddings-based retrieval systems using 
sentence-transformers, OpenAI embeddings, BGE, E5 or similar deployed to real users. 
Handled embedding drift, index refresh, retrieval-quality regression in production. 
Production experience with vector databases or hybrid search infrastructure including 
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS or similar. 
Strong Python code quality. Hands-on experience designing evaluation frameworks 
for ranking systems using NDCG, MRR, MAP, offline-to-online correlation, A/B test 
interpretation.

Nice to have: LLM fine-tuning experience with LoRA, QLoRA, PEFT. Learning-to-rank 
models with XGBoost or neural approaches. HR-tech recruiting tech marketplace products. 
Distributed systems large-scale inference optimization. Open-source contributions in 
AI ML space.

Disqualifiers: Pure research without production deployment. Only recent LangChain 
projects without pre-LLM ML experience. Only architecture or tech lead roles without 
recent production code. Title-chasers switching every 1.5 years. Only consulting firms 
TCS Infosys Wipro Accenture Cognizant Capgemini in entire career. Primary expertise in 
computer vision speech robotics without NLP IR exposure.

Location: Pune Noida preferred. Hybrid flexible. India candidates from Hyderabad Mumbai 
Delhi NCR welcome. Notice period sub-30 days preferred, can buy out up to 30 days.

Ideal candidate: 6-8 years total experience, 4-5 in applied ML AI at product companies. 
Shipped end-to-end ranking search recommendation system at meaningful scale. Strong 
opinions about retrieval evaluation LLM integration. Located in or willing to relocate 
to Noida or Pune. Active on Redrob platform.
"""

# ============================================================================
# Company Classifications
# ============================================================================
SERVICES_COMPANIES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl',
    'tech mahindra', 'mindtree', 'lti', 'genpact', 'wns', 'hexaware',
    'mphasis', 'persistent systems', 'l&t infotech', 'cyient',
    'tata consultancy services', 'cognizant technology solutions',
}

# Product companies recognized as strong signals
TIER1_PRODUCT_COMPANIES = {
    'google', 'meta', 'microsoft', 'apple', 'amazon', 'netflix', 'linkedin',
    'salesforce', 'adobe', 'stripe', 'uber', 'airbnb', 'spotify',
    'flipkart', 'razorpay', 'cred', 'swiggy', 'zomato', 'phonepe',
    'dream11', 'meesho', 'groww', 'zerodha', 'paytm', 'ola',
}

TIER2_PRODUCT_COMPANIES = {
    'freshworks', 'browserstack', 'postman', 'inmobi', 'sharechat',
    'nykaa', 'upgrad', "byju's", 'unacademy', 'lenskart',
    'observe.ai', 'yellow.ai', 'verloop.io', 'aganitha', 'niramai',
    'mad street den', 'krutrim', 'sarvam ai', 'fractal',
    # Fictional product companies in the dataset
    'pied piper', 'hooli', 'initech', 'wayne enterprises', 'stark industries',
    'globex inc', 'acme corp', 'dunder mifflin',
}

# ============================================================================
# Title Classifications
# ============================================================================

# Tier 1: Exact match to what the JD describes
TIER1_TITLES = {
    'senior ai engineer', 'lead ai engineer', 'ai engineer',
    'staff machine learning engineer', 'senior machine learning engineer',
    'machine learning engineer', 'applied ml engineer',
    'recommendation systems engineer', 'search engineer',
    'senior nlp engineer', 'nlp engineer',
    'senior applied scientist',
}

# Tier 2: Strong adjacent roles
TIER2_TITLES = {
    'senior software engineer (ml)', 'ai research engineer',
    'ai specialist', 'ml engineer',
    'senior data scientist', 'data scientist',
}

# Tier 3: Could be a fit with the right background
TIER3_TITLES = {
    'junior ml engineer', 'computer vision engineer',
    'senior software engineer', 'senior data engineer',
    'backend engineer', 'data engineer', 'analytics engineer',
}

# Tier 4: Weak fit but still technically possible
TIER4_TITLES = {
    'software engineer', 'full stack developer', 'cloud engineer',
    'devops engineer', 'java developer', 'frontend engineer',
    'mobile developer', 'data analyst', '.net developer', 'qa engineer',
}

# Everything else (non-tech) is filtered out
ALL_ALLOWED_TITLES = TIER1_TITLES | TIER2_TITLES | TIER3_TITLES | TIER4_TITLES

# Non-tech titles to explicitly filter out
NON_TECH_TITLES = {
    'business analyst', 'hr manager', 'mechanical engineer', 'accountant',
    'project manager', 'customer support', 'operations manager',
    'content writer', 'sales executive', 'civil engineer',
    'graphic designer', 'marketing manager',
}

# ============================================================================
# Skill Categories (mapped to scoring weights)
# ============================================================================

# Core Retrieval & Search — highest value for this role
SKILLS_CORE_RETRIEVAL = {
    'embeddings', 'vector search', 'semantic search', 'sentence transformers',
    'information retrieval', 'faiss', 'pinecone', 'milvus', 'qdrant',
    'weaviate', 'elasticsearch', 'opensearch', 'pgvector', 'bm25',
    'haystack', 'information retrieval systems', 'search infrastructure',
    'search backend', 'search & discovery', 'indexing algorithms',
    'text encoders', 'vector representations', 'content matching',
}

# Core Ranking — very high value
SKILLS_CORE_RANKING = {
    'learning to rank', 'recommendation systems', 'ranking systems',
}

# General AI/NLP/LLM — high value
SKILLS_GENERAL_AI = {
    'llms', 'rag', 'fine-tuning llms', 'prompt engineering',
    'hugging face transformers', 'langchain', 'llamaindex',
    'nlp', 'natural language processing',
    'pytorch', 'tensorflow', 'scikit-learn',
    'machine learning', 'deep learning',
    'lora', 'qlora', 'peft', 'model adaptation',
    'open-source ml libraries',
}

# ML Ops & Systems — moderate value
SKILLS_ML_SYSTEMS = {
    'python', 'mlops', 'mlflow', 'kubeflow', 'bentoml',
    'weights & biases', 'workflow orchestration',
    'data pipelines', 'airflow', 'kafka', 'spark',
    'apache beam', 'apache flink', 'databricks',
}

# Infrastructure — lower value but shows production awareness
SKILLS_INFRA = {
    'docker', 'kubernetes', 'aws', 'azure', 'gcp',
    'go', 'rust', 'fastapi', 'django', 'flask',
    'ci/cd', 'terraform', 'microservices',
}

# Feature Engineering & Data Science — moderate value
SKILLS_DATA_SCIENCE = {
    'data science', 'feature engineering', 'statistical modeling',
    'forecasting', 'time series', 'reinforcement learning',
    'dbt', 'bigquery', 'snowflake', 'etl', 'sql', 'postgresql',
}

# Computer Vision & Speech — low value per JD
SKILLS_CV_SPEECH = {
    'computer vision', 'cnn', 'object detection', 'image classification',
    'opencv', 'yolo', 'diffusion models', 'gans',
    'asr', 'speech recognition', 'tts',
}

# Non-tech skills — zero or negative value
SKILLS_NON_TECH = {
    'accounting', 'tally', 'sales', 'marketing', 'seo',
    'content writing', 'project management', 'scrum', 'agile',
    'figma', 'photoshop', 'illustrator', 'excel', 'powerpoint',
    'six sigma', 'sap', 'salesforce crm',
    'html', 'css', 'redux', 'react', 'angular', 'vue.js',
    'next.js', 'webpack', 'tailwind', 'typescript', 'javascript',
    'spring boot', 'java', 'node.js', '.net', 'graphql',
    'rest apis', 'redis', 'mongodb', 'grpc',
}

# Skill weight mapping (points per skill)
SKILL_WEIGHTS = {}
for s in SKILLS_CORE_RETRIEVAL:
    SKILL_WEIGHTS[s] = 10.0
for s in SKILLS_CORE_RANKING:
    SKILL_WEIGHTS[s] = 10.0
for s in SKILLS_GENERAL_AI:
    SKILL_WEIGHTS[s] = 6.0
for s in SKILLS_ML_SYSTEMS:
    SKILL_WEIGHTS[s] = 4.0
for s in SKILLS_INFRA:
    SKILL_WEIGHTS[s] = 2.0
for s in SKILLS_DATA_SCIENCE:
    SKILL_WEIGHTS[s] = 3.0
for s in SKILLS_CV_SPEECH:
    SKILL_WEIGHTS[s] = 1.0
for s in SKILLS_NON_TECH:
    SKILL_WEIGHTS[s] = 0.0

# Proficiency multipliers
PROFICIENCY_MULT = {
    'expert': 1.2,
    'advanced': 1.0,
    'intermediate': 0.7,
    'beginner': 0.3,
}

# ============================================================================
# Career Description Keywords (for keyword bonus scoring)
# ============================================================================
CAREER_KEYWORDS_HIGH = {
    'vector database': 3.0, 'hybrid search': 3.0, 'hybrid retrieval': 3.0,
    'ndcg': 3.0, 'mean average precision': 3.0, 'learning to rank': 3.0,
    'a/b test': 2.0, 'a/b testing': 2.0,
    'embedding drift': 3.0, 'retrieval quality': 3.0,
    'search ranking': 2.0, 'candidate ranking': 2.0,
    'relevance': 1.5, 'recall': 1.5, 'precision': 1.5,
}

CAREER_KEYWORDS_MEDIUM = {
    'rag': 2.0, 'retrieval augmented': 2.0,
    'recommendation system': 2.0, 'recommender': 2.0,
    'collaborative filtering': 2.0, 'content-based': 1.5,
    'embeddings': 1.5, 'sentence-transformer': 2.0,
    'pinecone': 1.5, 'milvus': 1.5, 'qdrant': 1.5,
    'weaviate': 1.5, 'faiss': 1.5, 'elasticsearch': 1.0,
    'opensearch': 1.0, 'semantic search': 2.0,
    'fine-tuning': 1.5, 'lora': 1.5,
}

CAREER_KEYWORDS_LOW = {
    'production': 0.5, 'deployed': 0.5, 'shipped': 0.5,
    'scale': 0.5, 'scaled': 0.5, 'million': 0.5,
    'latency': 0.5, 'real-time': 0.5, 'real time': 0.5,
    'ranking': 0.3, 'retrieval': 0.3, 'search': 0.3,
    'nlp': 0.3, 'nlu': 0.3, 'information retrieval': 0.5,
}

ALL_CAREER_KEYWORDS = {**CAREER_KEYWORDS_HIGH, **CAREER_KEYWORDS_MEDIUM, **CAREER_KEYWORDS_LOW}

# ============================================================================
# Location / Geography
# ============================================================================
PREFERRED_LOCATIONS_INDIA = {
    'noida', 'pune', 'delhi', 'delhi ncr', 'gurgaon', 'gurugram',
    'mumbai', 'hyderabad', 'bangalore', 'bengaluru',
}

OTHER_INDIA_CITIES = {
    'chennai', 'kolkata', 'kochi', 'trivandrum', 'ahmedabad',
    'indore', 'jaipur', 'bhubaneswar', 'chandigarh', 'lucknow',
    'coimbatore', 'vizag', 'nagpur', 'patna', 'bhopal',
}

# ============================================================================
# Behavioral Signal Thresholds
# ============================================================================
RECENCY_EXCELLENT_DAYS = 90
RECENCY_GOOD_DAYS = 180

RESPONSE_RATE_EXCELLENT = 0.75
RESPONSE_RATE_GOOD = 0.50
RESPONSE_RATE_POOR = 0.15

RESPONSE_TIME_FAST_HOURS = 24
RESPONSE_TIME_OK_HOURS = 72

NOTICE_PERIOD_IDEAL = 30
NOTICE_PERIOD_OK = 60
NOTICE_PERIOD_LONG = 90

GITHUB_ACTIVE_THRESHOLD = 40
