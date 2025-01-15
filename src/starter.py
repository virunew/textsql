from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import spacy
import yaml
from llmware.library import Library
from llmware.agents import LLMfx
from llmware.embeddings import EmbeddingChromaDB
from llmware.models import ModelCatalog
import chromadb
from chromadb.config import Settings
from llmware.configs import LLMWareConfig


@dataclass
class SemanticTerm:
    original: str
    embedding: np.ndarray
    synonyms: List[str] = None
    hypernyms: List[str] = None

@dataclass
class ColumnMapping:
    name: str
    description: str
    domain_terms: List[SemanticTerm]
    embedding: np.ndarray

@dataclass
class TableMapping:
    name: str
    description: str
    domain_terms: List[SemanticTerm]
    columns: Dict[str, ColumnMapping]
    embedding: np.ndarray

class SemanticAnalyzer:
    def __init__(self):
        # Create a new library for ChromaDB
        library = Library().create_new_library("text2sql")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.create_collection("embeddings")

        # Initialize LLMWare for embedding generation with the correct library name
        self.embedding_model = EmbeddingChromaDB(library=library)
        self.nlp = spacy.load('en_core_web_sm')
        self.similarity_threshold = 0.75

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text string using LLMWare
        """
        embedding = self.embedding_model.embed(text)
        # Store the text and its embedding in ChromaDB
        self.collection.add(documents=[text], embeddings=[embedding])
        return embedding

    def get_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def extract_query_intent(self, query: str) -> dict:
        """
        Extract semantic intent and key components from the query
        """
        doc = self.nlp(query)
        root_verb = next((token for token in doc if token.dep_ == 'ROOT'), None)
        entities = {ent.label_: ent.text for ent in doc.ents}
        conditions = []

        for token in doc:
            if token.like_num:
                left_token = token.nbor(-1)
                if left_token.text in ['>', '<', '>=', '<=', '=']:
                    conditions.append({
                        'operator': left_token.text,
                        'value': token.text,
                        'subject': left_token.head.text
                    })

        return {
            'primary_action': root_verb.text if root_verb else None,
            'entities': entities,
            'conditions': conditions,
            'doc': doc
        }

    def get_semantic_terms(self, text: str) -> SemanticTerm:
        """
        Create semantic term with embeddings and related terms
        """
        embedding = self.get_embedding(text)
        # Placeholder for synonyms and hypernyms
        synonyms = []  # You can implement synonym extraction if needed
        hypernyms = []  # You can implement hypernym extraction if needed

        return SemanticTerm(
            original=text,
            embedding=embedding,
            synonyms=synonyms,
            hypernyms=hypernyms
        )

class DomainConfig:
    def __init__(self, config_path: str, semantic_analyzer: SemanticAnalyzer):
        self.semantic_analyzer = semantic_analyzer
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tables = self._process_tables()
        self.term_embeddings = self._create_term_embeddings()

    def _process_tables(self) -> Dict[str, TableMapping]:
        tables = {}
        for table_name, table_info in self.config['tables'].items():
            table_terms = [
                self.semantic_analyzer.get_semantic_terms(term)
                for term in table_info['domain_terms']
            ]
            
            columns = {}
            for col_name, col_info in table_info['columns'].items():
                col_terms = [
                    self.semantic_analyzer.get_semantic_terms(term)
                    for term in col_info['domain_terms']
                ]
                
                columns[col_name] = ColumnMapping(
                    name=col_name,
                    description=col_info['description'],
                    domain_terms=col_terms,
                    embedding=self.semantic_analyzer.get_embedding(
                        col_info['description']
                    )
                )
            
            tables[table_name] = TableMapping(
                name=table_name,
                description=table_info['description'],
                domain_terms=table_terms,
                columns=columns,
                embedding=self.semantic_analyzer.get_embedding(
                    table_info['description']
                )
            )
        
        return tables

    def _create_term_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}
        for table in self.tables.values():
            for term in table.domain_terms:
                embeddings[term.original] = term.embedding
                
            for column in table.columns.values():
                for term in column.domain_terms:
                    embeddings[term.original] = term.embedding
        
        return embeddings

class SemanticQueryAnalyzer:
    def __init__(self, domain_config: DomainConfig, semantic_analyzer: SemanticAnalyzer):
        self.config = domain_config
        self.semantic_analyzer = semantic_analyzer
    
    def analyze_query(self, query: str) -> dict:
        intent = self.semantic_analyzer.extract_query_intent(query)
        query_embedding = self.semantic_analyzer.get_embedding(query)
        matched_elements = self._match_query_elements(query, query_embedding, intent)
        
        return {
            'intent': intent,
            'matched_elements': matched_elements,
            'query_embedding': query_embedding
        }
    
    def _match_query_elements(self, query: str, query_embedding: np.ndarray, intent: dict) -> dict:
        matches = {
            'tables': [],
            'columns': [],
            'conditions': []
        }
        
        for table_name, table in self.config.tables.items():
            similarity = self.semantic_analyzer.get_semantic_similarity(
                query_embedding, 
                table.embedding
            )
            
            if similarity > self.semantic_analyzer.similarity_threshold:
                matches['tables'].append({
                    'table_name': table_name,
                    'similarity': similarity,
                    'matched_term': None
                })
        
        doc = intent['doc']
        for token in doc:
            token_embedding = self.semantic_analyzer.get_embedding(token.text)
            for table_name, table in self.config.tables.items():
                for col_name, column in table.columns.items():
                    for term in column.domain_terms:
                        similarity = self.semantic_analyzer.get_semantic_similarity(
                            token_embedding, 
                            term.embedding
                        )
                        
                        if similarity > self.semantic_analyzer.similarity_threshold:
                            matches['columns'].append({
                                'table_name': table_name,
                                'column_name': col_name,
                                'similarity': similarity,
                                'matched_term': term.original,
                                'query_term': token.text
                            })
        
        matches['conditions'] = intent['conditions']
        
        return matches

class LLMQueryGenerator:
    def __init__(self, domain_config: DomainConfig, semantic_analyzer: SemanticAnalyzer, llm_client):
        self.config = domain_config
        self.semantic_analyzer = semantic_analyzer
        self.llm_client = llm_client
    
    def generate_prompt(self, query: str, analysis: dict) -> str:
        schema_context = self._get_schema_context(analysis['matched_elements'])
        intent_context = self._get_intent_context(analysis['intent'])
        
        prompt = f"""
        Given the following database schema and query analysis:
        
        Schema Information:
        {schema_context}
        
        Query Intent:
        {intent_context}
        
        Original Query: "{query}"
        
        Generate a SQL query that captures the semantic intent of the question.
        Use the exact table and column names from the schema.
        Include appropriate joins based on the relationships provided.
        
        Return only the SQL query without any explanations.
        """
        
        return prompt
    
    def _get_schema_context(self, matched_elements: dict) -> str:
        context = []
        
        if matched_elements['tables']:
            context.append("Relevant Tables:")
            for match in matched_elements['tables']:
                table = self.config.tables[match['table_name']]
                context.append(f"- {match['table_name']}: {table.description}")
        
        if matched_elements['columns']:
            context.append("\nRelevant Columns:")
            for match in matched_elements['columns']:
                column = self.config.tables[match['table_name']].columns[match['column_name']]
                context.append(
                    f"- {match['table_name']}.{match['column_name']}: {column.description}"
                    f" (matched term: '{match['matched_term']}' for '{match['query_term']}')"
                )
        
        return "\n".join(context)
    
    def _get_intent_context(self, intent: dict) -> str:
        context = []
        
        if intent['primary_action']:
            context.append(f"Primary Action: {intent['primary_action']}")
        
        if intent['entities']:
            context.append("\nIdentified Entities:")
            for entity_type, entity in intent['entities'].items():
                context.append(f"- {entity_type}: {entity}")
        
        if intent['conditions']:
            context.append("\nIdentified Conditions:")
            for condition in intent['conditions']:
                context.append(
                    f"- {condition['subject']} {condition['operator']} {condition['value']}"
                )
        
        return "\n".join(context)

class QueryTranslator:
    def __init__(self, config_path: str, llm_client):
        self.semantic_analyzer = SemanticAnalyzer()
        self.config = DomainConfig(config_path, self.semantic_analyzer)
        self.query_analyzer = SemanticQueryAnalyzer(self.config, self.semantic_analyzer)
        self.llm_generator = LLMQueryGenerator(
            self.config, 
            self.semantic_analyzer,
            llm_client
        )
    
    async def translate_to_sql(self, natural_query: str) -> str:
        analysis = self.query_analyzer.analyze_query(natural_query)
        prompt = self.llm_generator.generate_prompt(natural_query, analysis)
        sql_query = await self._get_llm_response(prompt)
        return sql_query
    
    async def _get_llm_response(self, prompt: str) -> str:
        response = await self.llm_client.generate(prompt)
        return response

# Example usage
async def main():


    # Set the active database to SQLite (or any other you prefer)
    LLMWareConfig().set_active_db("sqlite")

# Set the vector database to ChromaDB
    LLMWareConfig().set_vector_db("chromadb")
    llm_client = LLMfx()  # Initialize LLM client from llmware
    translator = QueryTranslator("domain_config.yaml", llm_client)
    
    queries = [
        "What's the typical creditworthiness of borrowers with payment issues?",
        "Show me clients whose financial reliability is concerning",
        "Which accounts have a history of falling behind on obligations?"
    ]
    
    for query in queries:
        sql = await translator.translate_to_sql(query)
        print(f"\nOriginal Query: {query}")
        print(f"Generated SQL:\n{sql}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())