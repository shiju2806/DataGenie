# backend/knowledge_framework.py - Hybrid Knowledge Architecture
import json
import os
import requests
import time
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import pickle
import hashlib
from functools import lru_cache
import pandas as pd
from mathematical_engine import MathematicalKnowledgeEngine

logger = logging.getLogger(__name__)


@dataclass
class ConceptDefinition:
    name: str
    definition: str
    category: str
    domain: str
    synonyms: List[str]
    related_concepts: List[str]
    source: str
    confidence: float
    last_updated: datetime
    calculation_method: Optional[str] = None
    regulatory_context: Optional[str] = None


@dataclass
class KnowledgeSource:
    name: str
    type: str  # 'local', 'free_api', 'paid_api'
    base_url: Optional[str]
    api_key: Optional[str]
    rate_limit: int
    cost_per_call: float
    reliability: float
    last_used: Optional[datetime] = None
    calls_today: int = 0


class LocalKnowledgeEngine:
    """Core local knowledge foundation - always available, 100% accurate"""

    def __init__(self, cache_dir: str = "cache/knowledge"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.core_concepts = {}
        self.industry_concepts = {}
        self.user_concepts = {}

        self._initialize_core_knowledge()
        self._load_cached_concepts()

    def _initialize_core_knowledge(self):
        """Initialize foundational business concepts"""

        # Universal business concepts
        self.core_concepts = {
            'financial': {
                'revenue': ConceptDefinition(
                    name='revenue',
                    definition='Total income generated from business operations',
                    category='financial',
                    domain='universal',
                    synonyms=['income', 'sales', 'earnings', 'receipts', 'turnover'],
                    related_concepts=['profit', 'cost', 'margin'],
                    source='local_core',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='sum_of_sales_transactions'
                ),
                'cost': ConceptDefinition(
                    name='cost',
                    definition='Expenses incurred in business operations',
                    category='financial',
                    domain='universal',
                    synonyms=['expense', 'expenditure', 'outlay', 'spending'],
                    related_concepts=['revenue', 'profit', 'budget'],
                    source='local_core',
                    confidence=1.0,
                    last_updated=datetime.now()
                ),
                'profit': ConceptDefinition(
                    name='profit',
                    definition='Revenue minus costs',
                    category='financial',
                    domain='universal',
                    synonyms=['earnings', 'income', 'margin', 'surplus'],
                    related_concepts=['revenue', 'cost', 'profitability'],
                    source='local_core',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='revenue - total_costs'
                )
            },
            'operational': {
                'efficiency': ConceptDefinition(
                    name='efficiency',
                    definition='Ratio of useful output to total input',
                    category='operational',
                    domain='universal',
                    synonyms=['productivity', 'performance', 'effectiveness'],
                    related_concepts=['optimization', 'utilization', 'waste'],
                    source='local_core',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='output / input'
                ),
                'volume': ConceptDefinition(
                    name='volume',
                    definition='Quantity or amount of business activity',
                    category='operational',
                    domain='universal',
                    synonyms=['quantity', 'amount', 'throughput', 'count'],
                    related_concepts=['capacity', 'demand', 'supply'],
                    source='local_core',
                    confidence=1.0,
                    last_updated=datetime.now()
                )
            }
        }

        # Initialize industry-specific knowledge
        self._initialize_industry_knowledge()

    def _initialize_industry_knowledge(self):
        """Initialize curated industry-specific concepts"""

        # Insurance industry knowledge
        self.industry_concepts['insurance'] = {
            'actuarial': {
                'mortality_rate': ConceptDefinition(
                    name='mortality_rate',
                    definition='Probability of death within a specified period',
                    category='actuarial',
                    domain='insurance',
                    synonyms=['death_rate', 'qx', 'mortality_probability'],
                    related_concepts=['life_expectancy', 'survival_rate', 'mortality_table'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='deaths / exposures',
                    regulatory_context='Required for statutory reserves calculation'
                ),
                'ae_ratio': ConceptDefinition(
                    name='ae_ratio',
                    definition='Actual to Expected ratio for experience analysis',
                    category='actuarial',
                    domain='insurance',
                    synonyms=['actual_expected_ratio', 'experience_ratio'],
                    related_concepts=['mortality_rate', 'expected_mortality', 'variance'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='actual_events / expected_events',
                    regulatory_context='Used in GAAP and statutory reporting'
                ),
                'lapse_rate': ConceptDefinition(
                    name='lapse_rate',
                    definition='Rate at which policies terminate voluntarily',
                    category='actuarial',
                    domain='insurance',
                    synonyms=['surrender_rate', 'termination_rate', 'discontinuance_rate'],
                    related_concepts=['persistency', 'retention', 'policy_duration'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='lapses / exposures'
                )
            },
            'regulatory': {
                'statutory_reserves': ConceptDefinition(
                    name='statutory_reserves',
                    definition='Reserves required by insurance regulators',
                    category='regulatory',
                    domain='insurance',
                    synonyms=['required_reserves', 'regulatory_reserves'],
                    related_concepts=['gaap_reserves', 'cash_value', 'liability'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    regulatory_context='NAIC statutory accounting principles'
                )
            }
        }

        # Banking industry knowledge
        self.industry_concepts['banking'] = {
            'credit_risk': {
                'default_rate': ConceptDefinition(
                    name='default_rate',
                    definition='Percentage of loans that fail to be repaid',
                    category='credit_risk',
                    domain='banking',
                    synonyms=['charge_off_rate', 'loss_rate', 'bad_debt_rate'],
                    related_concepts=['credit_quality', 'provision', 'recovery_rate'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='defaults / total_loans'
                ),
                'credit_score': ConceptDefinition(
                    name='credit_score',
                    definition='Numerical expression of creditworthiness',
                    category='credit_risk',
                    domain='banking',
                    synonyms=['fico_score', 'credit_rating', 'risk_score'],
                    related_concepts=['default_probability', 'credit_bureau', 'underwriting'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now()
                )
            },
            'regulatory': {
                'tier1_capital': ConceptDefinition(
                    name='tier1_capital',
                    definition='Core capital measure under Basel framework',
                    category='regulatory',
                    domain='banking',
                    synonyms=['core_capital', 'primary_capital'],
                    related_concepts=['capital_adequacy', 'basel', 'risk_weighted_assets'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    regulatory_context='Basel III capital requirements'
                )
            }
        }

        # Technology industry knowledge
        self.industry_concepts['technology'] = {
            'performance': {
                'latency': ConceptDefinition(
                    name='latency',
                    definition='Time delay in system response',
                    category='performance',
                    domain='technology',
                    synonyms=['response_time', 'delay', 'lag'],
                    related_concepts=['throughput', 'performance', 'sla'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='response_end_time - request_start_time'
                ),
                'uptime': ConceptDefinition(
                    name='uptime',
                    definition='Percentage of time system is operational',
                    category='performance',
                    domain='technology',
                    synonyms=['availability', 'reliability', 'service_level'],
                    related_concepts=['downtime', 'sla', 'maintenance'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now(),
                    calculation_method='operational_time / total_time'
                )
            },
            'user_metrics': {
                'dau': ConceptDefinition(
                    name='daily_active_users',
                    definition='Number of unique users active in a day',
                    category='user_metrics',
                    domain='technology',
                    synonyms=['dau', 'daily_users', 'active_users'],
                    related_concepts=['mau', 'engagement', 'retention'],
                    source='local_industry',
                    confidence=1.0,
                    last_updated=datetime.now()
                )
            }
        }

    def _load_cached_concepts(self):
        """Load previously learned concepts from cache"""
        cache_file = os.path.join(self.cache_dir, 'learned_concepts.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.user_concepts = cached_data.get('user_concepts', {})
                    logger.info(f"Loaded {len(self.user_concepts)} cached concepts")
            except Exception as e:
                logger.warning(f"Failed to load cached concepts: {e}")

    def save_concepts_cache(self):
        """Save learned concepts to cache"""
        cache_file = os.path.join(self.cache_dir, 'learned_concepts.pkl')
        try:
            cache_data = {
                'user_concepts': self.user_concepts,
                'last_updated': datetime.now()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Failed to save concepts cache: {e}")

    def find_concept(self, term: str, domain: str = None) -> Optional[ConceptDefinition]:
        """Find concept in local knowledge base"""
        term_lower = term.lower().strip()

        # Search priority: domain-specific -> core -> user concepts
        search_order = []

        if domain:
            search_order.append(self.industry_concepts.get(domain, {}))
        search_order.extend([self.core_concepts, self.user_concepts])

        for concept_dict in search_order:
            for category, concepts in concept_dict.items():
                for concept_name, concept_def in concepts.items():
                    # Direct name match
                    if concept_name.lower() == term_lower:
                        return concept_def

                    # Synonym match
                    if term_lower in [syn.lower() for syn in concept_def.synonyms]:
                        return concept_def

                    # Partial match in name or synonyms
                    if (term_lower in concept_name.lower() or
                            any(term_lower in syn.lower() for syn in concept_def.synonyms)):
                        return concept_def

        return None

    def add_concept(self, concept: ConceptDefinition, category: str = 'user_defined'):
        """Add new concept to local knowledge base"""
        if category not in self.user_concepts:
            self.user_concepts[category] = {}

        self.user_concepts[category][concept.name] = concept
        self.save_concepts_cache()
        logger.info(f"Added concept: {concept.name} to {category}")

    def get_related_concepts(self, concept_name: str, max_depth: int = 2) -> List[ConceptDefinition]:
        """Get related concepts using relationship graph"""
        related = []
        visited = set()

        def _find_related(name: str, depth: int):
            if depth > max_depth or name in visited:
                return

            visited.add(name)
            concept = self.find_concept(name)

            if concept:
                related.append(concept)
                for related_name in concept.related_concepts:
                    _find_related(related_name, depth + 1)

        _find_related(concept_name, 0)
        return related[1:]  # Exclude the original concept


class APIKnowledgeEngine:
    """Dynamic knowledge acquisition from external APIs"""

    def __init__(self):
        self.sources = self._initialize_knowledge_sources()
        self.cache_dir = "cache/api_knowledge"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.daily_usage = {}
        self._load_usage_stats()

    def _initialize_knowledge_sources(self) -> Dict[str, KnowledgeSource]:
        """Initialize available knowledge sources"""
        return {
            'wikidata': KnowledgeSource(
                name='Wikidata Query Service',
                type='free_api',
                base_url='https://query.wikidata.org/sparql',
                api_key=None,
                rate_limit=10000,  # Per day
                cost_per_call=0.0,
                reliability=0.85
            ),
            'sec_edgar': KnowledgeSource(
                name='SEC EDGAR API',
                type='free_api',
                base_url='https://data.sec.gov/api',
                api_key=None,
                rate_limit=10,  # Per second
                cost_per_call=0.0,
                reliability=0.95
            ),
            'federal_reserve': KnowledgeSource(
                name='Federal Reserve Economic Data',
                type='free_api',
                base_url='https://api.stlouisfed.org/fred',
                api_key=None,  # Requires free registration
                rate_limit=1000,  # Per day
                cost_per_call=0.0,
                reliability=0.98
            ),
            'arxiv': KnowledgeSource(
                name='arXiv API',
                type='free_api',
                base_url='http://export.arxiv.org/api',
                api_key=None,
                rate_limit=1000,  # Per day
                cost_per_call=0.0,
                reliability=0.90
            )
        }

    def _load_usage_stats(self):
        """Load daily API usage statistics"""
        usage_file = os.path.join(self.cache_dir, 'usage_stats.json')
        if os.path.exists(usage_file):
            try:
                with open(usage_file, 'r') as f:
                    self.daily_usage = json.load(f)
            except:
                self.daily_usage = {}

    def _save_usage_stats(self):
        """Save API usage statistics"""
        usage_file = os.path.join(self.cache_dir, 'usage_stats.json')
        try:
            with open(usage_file, 'w') as f:
                json.dump(self.daily_usage, f)
        except Exception as e:
            logger.error(f"Failed to save usage stats: {e}")

    def _can_use_source(self, source_name: str) -> bool:
        """Check if we can use an API source without exceeding limits"""
        source = self.sources.get(source_name)
        if not source:
            return False

        today = datetime.now().strftime('%Y-%m-%d')
        usage_key = f"{source_name}_{today}"

        current_usage = self.daily_usage.get(usage_key, 0)
        return current_usage < source.rate_limit

    def _record_api_call(self, source_name: str):
        """Record API call for rate limiting"""
        today = datetime.now().strftime('%Y-%m-%d')
        usage_key = f"{source_name}_{today}"

        self.daily_usage[usage_key] = self.daily_usage.get(usage_key, 0) + 1
        self.sources[source_name].calls_today += 1
        self.sources[source_name].last_used = datetime.now()

        self._save_usage_stats()

    @lru_cache(maxsize=1000)
    def query_wikidata(self, concept: str, domain: str = None) -> Optional[ConceptDefinition]:
        """Query Wikidata for concept definition"""
        if not self._can_use_source('wikidata'):
            return None

        try:
            # Simple SPARQL query for concept definition
            sparql_query = f"""
            SELECT ?item ?itemLabel ?itemDescription WHERE {{
              ?item rdfs:label "{concept}"@en .
              ?item schema:description ?itemDescription .
              FILTER(LANG(?itemDescription) = "en")
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
            }}
            LIMIT 1
            """

            response = requests.get(
                self.sources['wikidata'].base_url,
                params={'query': sparql_query, 'format': 'json'},
                timeout=10
            )

            self._record_api_call('wikidata')

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])

                if results:
                    result = results[0]
                    definition = result.get('itemDescription', {}).get('value', '')

                    return ConceptDefinition(
                        name=concept,
                        definition=definition,
                        category='general',
                        domain=domain or 'general',
                        synonyms=[],
                        related_concepts=[],
                        source='wikidata',
                        confidence=0.85,
                        last_updated=datetime.now()
                    )

        except Exception as e:
            logger.warning(f"Wikidata query failed for '{concept}': {e}")

        return None

    def query_sec_edgar(self, concept: str) -> Optional[ConceptDefinition]:
        """Query SEC EDGAR for financial/regulatory definitions"""
        if not self._can_use_source('sec_edgar'):
            return None

        # This would implement SEC API querying for financial terms
        # For now, return None to stay within scope
        return None

    def search_arxiv(self, concept: str, domain: str = None) -> Optional[ConceptDefinition]:
        """Search arXiv for academic definitions"""
        if not self._can_use_source('arxiv'):
            return None

        try:
            # Search for papers mentioning the concept
            query = f"all:{concept}"
            if domain:
                query += f" AND cat:{domain}*"

            response = requests.get(
                f"{self.sources['arxiv'].base_url}/query",
                params={
                    'search_query': query,
                    'start': 0,
                    'max_results': 5
                },
                timeout=15
            )

            self._record_api_call('arxiv')

            if response.status_code == 200:
                # Parse XML response (simplified)
                # In reality, you'd parse the arXiv XML format
                # and extract definitions from abstracts

                return ConceptDefinition(
                    name=concept,
                    definition=f"Academic definition of {concept} (from arXiv research)",
                    category='academic',
                    domain=domain or 'research',
                    synonyms=[],
                    related_concepts=[],
                    source='arxiv',
                    confidence=0.80,
                    last_updated=datetime.now()
                )

        except Exception as e:
            logger.warning(f"arXiv search failed for '{concept}': {e}")

        return None


class HybridKnowledgeFramework:
    """Main knowledge framework that orchestrates local and API knowledge"""

    def __init__(self, mathematical_engine: MathematicalKnowledgeEngine):
        self.mathematical_engine = mathematical_engine
        self.local_engine = LocalKnowledgeEngine()
        self.api_engine = APIKnowledgeEngine()

        self.concept_cache = {}
        self.query_history = []

        logger.info("🧠 Hybrid Knowledge Framework initialized")
        logger.info(f"   📚 Local concepts loaded: {self._count_local_concepts()}")
        logger.info(f"   🌐 API sources available: {len(self.api_engine.sources)}")

    def _count_local_concepts(self) -> int:
        """Count total local concepts"""
        count = 0
        for category in self.local_engine.core_concepts.values():
            count += len(category)
        for industry in self.local_engine.industry_concepts.values():
            for category in industry.values():
                count += len(category)
        return count

    def find_concept(self, term: str, domain: str = None,
                     use_apis: bool = True) -> Optional[ConceptDefinition]:
        """Find concept using hybrid approach: local first, then APIs"""

        # Record query for learning
        self.query_history.append({
            'term': term,
            'domain': domain,
            'timestamp': datetime.now()
        })

        # Check cache first
        cache_key = f"{term}_{domain or 'general'}".lower()
        if cache_key in self.concept_cache:
            cached_concept = self.concept_cache[cache_key]
            # Check if cache is still fresh (24 hours)
            if (datetime.now() - cached_concept.last_updated).days < 1:
                return cached_concept

        # 1. Try local knowledge first (instant, 100% reliable)
        concept = self.local_engine.find_concept(term, domain)
        if concept:
            self.concept_cache[cache_key] = concept
            return concept

        # 2. If not found locally and APIs are enabled, try external sources
        if use_apis:
            # Try APIs in order of reliability and relevance
            api_sources = [
                ('wikidata', self.api_engine.query_wikidata),
                ('arxiv', self.api_engine.search_arxiv)
            ]

            for source_name, query_func in api_sources:
                if self.api_engine._can_use_source(source_name):
                    try:
                        concept = query_func(term, domain)
                        if concept:
                            # Cache the result
                            self.concept_cache[cache_key] = concept

                            # Auto-save high-confidence concepts to local knowledge
                            if concept.confidence > 0.9:
                                self.local_engine.add_concept(concept, 'api_verified')

                            return concept
                    except Exception as e:
                        logger.warning(f"API query failed for {source_name}: {e}")
                        continue

        return None

    def enhance_query_with_concepts(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Enhance user query with concept knowledge"""

        # Extract potential concepts from query
        query_words = query.lower().split()
        found_concepts = []
        enhanced_understanding = {}

        # Look for mathematical concepts first
        mathematical_terms = self._extract_mathematical_terms(query)
        for term in mathematical_terms:
            enhanced_understanding[term] = {
                'type': 'mathematical',
                'confidence': 1.0,
                'source': 'mathematical_engine'
            }

        # Look for business/domain concepts
        for word in query_words:
            if len(word) > 3:  # Skip short words
                concept = self.find_concept(word, domain, use_apis=False)  # Use local only for speed
                if concept:
                    found_concepts.append(concept)
                    enhanced_understanding[word] = {
                        'type': 'domain_concept',
                        'definition': concept.definition,
                        'synonyms': concept.synonyms,
                        'confidence': concept.confidence,
                        'source': concept.source
                    }

        # Generate query enhancement suggestions
        enhancement_suggestions = self._generate_query_enhancements(query, found_concepts, domain)

        return {
            'original_query': query,
            'found_concepts': [c.name for c in found_concepts],
            'enhanced_understanding': enhanced_understanding,
            'suggestions': enhancement_suggestions,
            'confidence': self._calculate_understanding_confidence(enhanced_understanding)
        }

    def _extract_mathematical_terms(self, query: str) -> List[str]:
        """Extract mathematical/statistical terms from query"""
        mathematical_keywords = [
            'mean', 'average', 'median', 'mode', 'std', 'variance', 'correlation',
            'regression', 'trend', 'distribution', 'probability', 'ratio', 'rate',
            'percentage', 'percent', 'sum', 'total', 'count', 'frequency',
            'significant', 'confidence', 'interval', 'test', 'analysis'
        ]

        query_lower = query.lower()
        found_terms = []

        for term in mathematical_keywords:
            if term in query_lower:
                found_terms.append(term)

        return found_terms

    def _generate_query_enhancements(self, query: str, concepts: List[ConceptDefinition],
                                     domain: str = None) -> List[str]:
        """Generate suggestions to enhance query understanding"""
        suggestions = []

        # Suggest related concepts
        for concept in concepts:
            for related in concept.related_concepts[:2]:  # Top 2 related concepts
                related_concept = self.find_concept(related, domain, use_apis=False)
                if related_concept:
                    suggestions.append(f"Consider also analyzing {related} (related to {concept.name})")

        # Suggest mathematical enhancements
        mathematical_terms = self._extract_mathematical_terms(query)
        if mathematical_terms:
            if 'correlation' in mathematical_terms:
                suggestions.append("For correlation analysis, consider both Pearson and Spearman methods")
            if any(term in mathematical_terms for term in ['mean', 'average']):
                suggestions.append("Consider also median and mode for complete central tendency analysis")
            if 'trend' in mathematical_terms:
                suggestions.append("Time series analysis might provide deeper trend insights")

        # Domain-specific suggestions
        if domain == 'insurance':
            if any(term in query.lower() for term in ['mortality', 'death']):
                suggestions.append("Consider A/E ratio analysis for mortality experience evaluation")
        elif domain == 'banking':
            if any(term in query.lower() for term in ['credit', 'default']):
                suggestions.append("Consider segmenting by credit score or loan type")

        return suggestions[:3]  # Limit to top 3 suggestions

    def _calculate_understanding_confidence(self, enhanced_understanding: Dict[str, Any]) -> float:
        """Calculate confidence in query understanding"""
        if not enhanced_understanding:
            return 0.3  # Low confidence if no concepts recognized

        total_confidence = 0
        for concept_info in enhanced_understanding.values():
            total_confidence += concept_info.get('confidence', 0.5)

        average_confidence = total_confidence / len(enhanced_understanding)

        # Boost confidence if we have mathematical terms (well understood)
        mathematical_count = sum(1 for info in enhanced_understanding.values()
                                 if info.get('type') == 'mathematical')
        if mathematical_count > 0:
            average_confidence += 0.1

        return min(1.0, average_confidence)

    def suggest_analysis_methods(self, query: str, data_characteristics: Dict[str, Any],
                                 domain: str = None) -> List[Dict[str, Any]]:
        """Suggest appropriate analysis methods based on query and data"""

        # Enhanced query understanding
        query_enhancement = self.enhance_query_with_concepts(query, domain)

        # Get mathematical method recommendations
        mathematical_terms = self._extract_mathematical_terms(query)
        analysis_intent = ' '.join(mathematical_terms) if mathematical_terms else query

        # Convert data characteristics to format expected by mathematical engine
        math_data_chars = self._convert_data_characteristics(data_characteristics)

        # Get method recommendations from mathematical engine
        variables = data_characteristics.get('numeric_columns', [])
        if not variables:
            variables = data_characteristics.get('columns', [])[:2]  # Take first 2 columns as fallback

        try:
            method_recommendations = self.mathematical_engine.recommend_methods(
                analysis_intent, math_data_chars, variables
            )
        except:
            method_recommendations = []

        # Combine recommendations with concept knowledge
        suggestions = []

        for method, confidence in method_recommendations:
            suggestion = {
                'method': method.name,
                'description': method.description,
                'confidence': confidence,
                'assumptions': method.assumptions,
                'type': 'mathematical'
            }
            suggestions.append(suggestion)

        # Add domain-specific suggestions
        domain_suggestions = self._get_domain_specific_suggestions(query, domain, data_characteristics)
        suggestions.extend(domain_suggestions)

        # Add concept-driven suggestions
        concept_suggestions = self._get_concept_driven_suggestions(query_enhancement, data_characteristics)
        suggestions.extend(concept_suggestions)

        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return suggestions[:8]

    def _convert_data_characteristics(self, data_characteristics: Dict[str, Any]) -> Any:
        """Convert data characteristics to mathematical engine format"""
        # This is a simplified conversion - in practice, you'd need to match
        # the exact DataCharacteristics format expected by mathematical engine
        from mathematical_engine import DataCharacteristics, DataType

        # Create mock data characteristics for the mathematical engine
        return DataCharacteristics(
            data_types={col: DataType.NUMERIC_CONTINUOUS for col in data_characteristics.get('numeric_columns', [])},
            sample_size=data_characteristics.get('rows', 100),
            missing_data={},
            distribution_properties={}
        )

    def _get_domain_specific_suggestions(self, query: str, domain: str,
                                         data_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get domain-specific analysis suggestions"""
        suggestions = []

        if domain == 'insurance':
            if 'mortality' in query.lower():
                suggestions.append({
                    'method': 'ae_ratio_analysis',
                    'description': 'Actual vs Expected mortality analysis',
                    'confidence': 0.95,
                    'type': 'domain_specific',
                    'calculation': 'actual_deaths / expected_deaths'
                })

            if 'lapse' in query.lower():
                suggestions.append({
                    'method': 'lapse_rate_analysis',
                    'description': 'Policy lapse rate calculation by duration',
                    'confidence': 0.90,
                    'type': 'domain_specific',
                    'calculation': 'lapses / exposures'
                })

        elif domain == 'banking':
            if any(term in query.lower() for term in ['credit', 'default', 'risk']):
                suggestions.append({
                    'method': 'credit_risk_analysis',
                    'description': 'Credit default rate and risk assessment',
                    'confidence': 0.90,
                    'type': 'domain_specific',
                    'calculation': 'defaults / total_loans'
                })

        elif domain == 'technology':
            if any(term in query.lower() for term in ['performance', 'latency', 'uptime']):
                suggestions.append({
                    'method': 'performance_analysis',
                    'description': 'System performance metrics analysis',
                    'confidence': 0.85,
                    'type': 'domain_specific'
                })

        return suggestions

    def _get_concept_driven_suggestions(self, query_enhancement: Dict[str, Any],
                                        data_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions based on identified concepts"""
        suggestions = []

        enhanced_understanding = query_enhancement.get('enhanced_understanding', {})

        for term, concept_info in enhanced_understanding.items():
            if concept_info.get('type') == 'domain_concept':
                synonyms = concept_info.get('synonyms', [])
                if synonyms:
                    suggestions.append({
                        'method': f'{term}_analysis',
                        'description': f'Analysis considering {term} and related terms: {", ".join(synonyms[:3])}',
                        'confidence': concept_info.get('confidence', 0.7),
                        'type': 'concept_driven'
                    })

        return suggestions

    def learn_from_user_feedback(self, query: str, selected_method: str,
                                 user_satisfaction: float, domain: str = None):
        """Learn from user feedback to improve recommendations"""

        feedback_entry = {
            'query': query,
            'domain': domain,
            'selected_method': selected_method,
            'satisfaction': user_satisfaction,
            'timestamp': datetime.now()
        }

        # Save feedback for future learning
        feedback_file = os.path.join(self.local_engine.cache_dir, 'user_feedback.json')

        try:
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
            else:
                feedback_data = []

            feedback_data.append(feedback_entry)

            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f)

            logger.info(f"Recorded user feedback: {user_satisfaction} for {selected_method}")

        except Exception as e:
            logger.error(f"Failed to save user feedback: {e}")

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of available knowledge"""

        local_count = self._count_local_concepts()
        api_status = {name: source.reliability for name, source in self.api_engine.sources.items()}

        return {
            'local_concepts': local_count,
            'api_sources': api_status,
            'cache_size': len(self.concept_cache),
            'recent_queries': len([q for q in self.query_history
                                   if (datetime.now() - q['timestamp']).days < 1]),
            'supported_domains': list(self.local_engine.industry_concepts.keys()),
            'mathematical_capabilities': {
                'descriptive_statistics': True,
                'inferential_statistics': True,
                'correlation_analysis': True,
                'regression_analysis': True,
                'hypothesis_testing': True
            }
        }

    def validate_concept_quality(self, concept: ConceptDefinition) -> Dict[str, Any]:
        """Validate the quality and reliability of a concept"""

        quality_score = 0.0
        issues = []

        # Check definition quality
        if len(concept.definition) > 20:
            quality_score += 0.3
        else:
            issues.append("Definition too short")

        # Check source reliability
        source_scores = {
            'local_core': 1.0,
            'local_industry': 1.0,
            'wikidata': 0.85,
            'arxiv': 0.90,
            'user_defined': 0.6
        }
        quality_score += source_scores.get(concept.source, 0.5) * 0.4

        # Check synonym coverage
        if len(concept.synonyms) >= 2:
            quality_score += 0.15
        elif len(concept.synonyms) >= 1:
            quality_score += 0.1
        else:
            issues.append("Limited synonym coverage")

        # Check relationship mapping
        if len(concept.related_concepts) >= 2:
            quality_score += 0.15
        elif len(concept.related_concepts) >= 1:
            quality_score += 0.1
        else:
            issues.append("Limited relationship mapping")

        return {
            'quality_score': min(1.0, quality_score),
            'issues': issues,
            'confidence': concept.confidence,
            'recommendation': 'approved' if quality_score > 0.8 else 'review_needed'
        }


# Integration function for existing codebase
def create_enhanced_adaptive_interpreter(openai_api_key: Optional[str] = None):
    """Create enhanced interpreter with hybrid knowledge framework"""

    # Initialize components
    mathematical_engine = MathematicalKnowledgeEngine()
    knowledge_framework = HybridKnowledgeFramework(mathematical_engine)

    # Import and enhance your existing adaptive interpreter
    try:
        from adaptive_interpreter import AdaptiveQueryInterpreter

        class EnhancedAdaptiveInterpreter(AdaptiveQueryInterpreter):
            def __init__(self, openai_api_key: Optional[str] = None):
                super().__init__(openai_api_key)
                self.knowledge_framework = knowledge_framework
                self.mathematical_engine = mathematical_engine

            def parse(self, query: str, target_dataset: Optional[str] = None) -> Dict[str, Any]:
                """Enhanced parsing with hybrid knowledge"""

                # Get domain from current profile
                domain = self.current_profile.get('primary_domain', 'general') if self.current_profile else None

                # Enhance query understanding with concepts
                query_enhancement = self.knowledge_framework.enhance_query_with_concepts(query, domain)

                # Use original parsing logic but enhanced with concepts
                original_result = super().parse(query, target_dataset)

                # Enhance the result with concept knowledge
                if original_result.get('confidence', 0) < 0.8:
                    # If confidence is low, try to improve with concept knowledge
                    concept_suggestions = self.knowledge_framework.suggest_analysis_methods(
                        query,
                        self.current_profile or {},
                        domain
                    )

                    original_result['concept_enhancements'] = query_enhancement
                    original_result['suggested_methods'] = concept_suggestions

                    # Boost confidence if we have good concept understanding
                    if query_enhancement.get('confidence', 0) > 0.7:
                        original_result['confidence'] = min(0.9, original_result.get('confidence', 0) + 0.2)

                return original_result

        return EnhancedAdaptiveInterpreter(openai_api_key), knowledge_framework, mathematical_engine

    except ImportError:
        # Fallback if adaptive interpreter not available
        logger.warning("Adaptive interpreter not found, using knowledge framework directly")
        return None, knowledge_framework, mathematical_engine


if __name__ == "__main__":
    # Test the hybrid knowledge framework
    mathematical_engine = MathematicalKnowledgeEngine()
    knowledge_framework = HybridKnowledgeFramework(mathematical_engine)

    # Test concept finding
    test_terms = ['mortality', 'correlation', 'revenue', 'default_rate', 'latency']

    print("🧪 Testing Hybrid Knowledge Framework")
    print("=" * 50)

    for term in test_terms:
        concept = knowledge_framework.find_concept(term, domain='insurance')
        if concept:
            print(f"✅ Found: {concept.name}")
            print(f"   Definition: {concept.definition}")
            print(f"   Source: {concept.source} (confidence: {concept.confidence})")
            print(f"   Synonyms: {', '.join(concept.synonyms[:3])}")
        else:
            print(f"❌ Not found: {term}")
        print()

    # Test query enhancement
    test_queries = [
        "Show me mortality trends by product type",
        "What's the correlation between age and premium?",
        "Analyze default rates by credit score"
    ]

    print("🔍 Testing Query Enhancement")
    print("-" * 30)

    for query in test_queries:
        enhancement = knowledge_framework.enhance_query_with_concepts(query, 'insurance')
        print(f"Query: {query}")
        print(f"Concepts found: {enhancement['found_concepts']}")
        print(f"Confidence: {enhancement['confidence']:.2f}")
        print(f"Suggestions: {enhancement['suggestions'][:2]}")
        print()

    # Test analysis method suggestions
    mock_data_characteristics = {
        'rows': 1000,
        'columns': ['age', 'premium', 'mortality_flag'],
        'numeric_columns': ['age', 'premium'],
        'categorical_columns': ['mortality_flag']
    }

    suggestions = knowledge_framework.suggest_analysis_methods(
        "Show me mortality correlation with age",
        mock_data_characteristics,
        'insurance'
    )

    print("💡 Analysis Method Suggestions")
    print("-" * 30)
    for suggestion in suggestions[:3]:
        print(f"Method: {suggestion['method']}")
        print(f"Description: {suggestion['description']}")
        print(f"Confidence: {suggestion.get('confidence', 0):.2f}")
        print(f"Type: {suggestion.get('type', 'unknown')}")
        print()

    # Show knowledge summary
    summary = knowledge_framework.get_knowledge_summary()
    print("📊 Knowledge Framework Summary")
    print("-" * 30)
    print(f"Local concepts: {summary['local_concepts']}")
    print(f"Supported domains: {', '.join(summary['supported_domains'])}")
    print(f"Mathematical capabilities: {len(summary['mathematical_capabilities'])}")
    print(f"API sources: {len(summary['api_sources'])}")

    print("\n✅ Hybrid Knowledge Framework test completed!")