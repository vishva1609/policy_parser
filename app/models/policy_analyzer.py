"""
Policy Analyzer - Extracts key policy statements and requirements from documents
"""
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class PolicyCategory(Enum):
    """Categories of policy statements"""
    ACCESS_CONTROL = "Access Control"
    DATA_SECURITY = "Data Security"
    COMPLIANCE = "Compliance"
    PRIVACY = "Privacy"
    INCIDENT_RESPONSE = "Incident Response"
    RISK_MANAGEMENT = "Risk Management"
    AUTHENTICATION = "Authentication"
    ENCRYPTION = "Encryption"
    AUDIT = "Audit & Monitoring"
    TRAINING = "Training & Awareness"
    VENDOR_MANAGEMENT = "Vendor Management"
    PHYSICAL_SECURITY = "Physical Security"
    NETWORK_SECURITY = "Network Security"
    GENERAL = "General"


@dataclass
class PolicyStatement:
    """Represents a single policy statement"""
    text: str
    category: PolicyCategory
    section: str
    page: int
    is_requirement: bool  # True if statement is mandatory (must/shall)
    confidence: float  # Confidence score for categorization


class PolicyAnalyzer:
    """Analyzes policy documents to extract key statements and requirements"""
    
    # Keywords for identifying requirements
    REQUIREMENT_KEYWORDS = [
        r'\bmust\b', r'\bshall\b', r'\brequired\b', r'\bmandatory\b',
        r'\bwill\b', r'\bneed to\b', r'\bhave to\b', r'\bshould\b'
    ]
    
    # Category keywords for classification
    CATEGORY_KEYWORDS = {
        PolicyCategory.ACCESS_CONTROL: [
            'access', 'authorization', 'permission', 'privilege', 'role',
            'user rights', 'access control', 'least privilege'
        ],
        PolicyCategory.DATA_SECURITY: [
            'data protection', 'data security', 'confidential', 'sensitive data',
            'data classification', 'information security'
        ],
        PolicyCategory.COMPLIANCE: [
            'compliance', 'regulatory', 'legal', 'standard', 'framework',
            'ISO', 'GDPR', 'HIPAA', 'SOC', 'regulation'
        ],
        PolicyCategory.PRIVACY: [
            'privacy', 'personal information', 'PII', 'data subject',
            'consent', 'anonymization', 'pseudonymization'
        ],
        PolicyCategory.INCIDENT_RESPONSE: [
            'incident', 'breach', 'response', 'emergency', 'disaster recovery',
            'business continuity', 'incident management'
        ],
        PolicyCategory.RISK_MANAGEMENT: [
            'risk', 'threat', 'vulnerability', 'risk assessment',
            'risk mitigation', 'risk management'
        ],
        PolicyCategory.AUTHENTICATION: [
            'authentication', 'password', 'MFA', 'multi-factor',
            'two-factor', 'login', 'credential', 'identity'
        ],
        PolicyCategory.ENCRYPTION: [
            'encryption', 'cryptography', 'TLS', 'SSL', 'encrypted',
            'cipher', 'key management'
        ],
        PolicyCategory.AUDIT: [
            'audit', 'logging', 'monitoring', 'tracking', 'log',
            'audit trail', 'surveillance'
        ],
        PolicyCategory.TRAINING: [
            'training', 'awareness', 'education', 'onboarding',
            'security awareness', 'phishing'
        ],
        PolicyCategory.VENDOR_MANAGEMENT: [
            'vendor', 'third party', 'supplier', 'contractor',
            'outsourcing', 'service provider'
        ],
        PolicyCategory.PHYSICAL_SECURITY: [
            'physical security', 'facility', 'premises', 'building',
            'access badge', 'CCTV'
        ],
        PolicyCategory.NETWORK_SECURITY: [
            'network', 'firewall', 'VPN', 'perimeter', 'DMZ',
            'intrusion detection', 'IDS', 'IPS'
        ]
    }
    
    def __init__(self):
        """Initialize the policy analyzer"""
        self.requirement_pattern = re.compile(
            '|'.join(self.REQUIREMENT_KEYWORDS),
            re.IGNORECASE
        )
    
    def analyze_document(self, chunks: List) -> List[PolicyStatement]:
        """
        Analyze document chunks and extract policy statements

        Args:
            chunks: List of DocumentChunk objects from DocumentPipeline

        Returns:
            List of PolicyStatement objects
        """
        statements = []

        for chunk in chunks:
            # Access chunk attributes (chunks are Pydantic models)
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            chunk_pages = chunk.pages if hasattr(chunk, 'pages') else [chunk.metadata.get('page', 0)]
            chunk_section = chunk.parent_context[0] if hasattr(chunk, 'parent_context') and chunk.parent_context else 'Unknown'

            # Skip non-content chunks (cover pages, TOC, index)
            if self._is_non_content(chunk_text):
                continue

            # Get first page number
            page_num = chunk_pages[0] if chunk_pages else 0
            
            # Split chunk into sentences
            sentences = self._split_into_sentences(chunk_text)
            
            for sentence in sentences:
                # Check if sentence is a policy statement
                if self._is_policy_statement(sentence):
                    category, confidence = self._categorize_statement(sentence)
                    is_requirement = self._is_requirement(sentence)
                    
                    statement = PolicyStatement(
                        text=sentence.strip(),
                        category=category,
                        section=chunk_section,
                        page=page_num,
                        is_requirement=is_requirement,
                        confidence=confidence
                    )
                    statements.append(statement)
        
        return statements

    def _is_non_content(self, text: str) -> bool:
        """
        Detect non-content text like table of contents, cover pages, and index pages.

        Heuristics:
        - High ratio of very short lines (typical of TOC: section titles + page numbers)
        - Many standalone numbers (page numbers in TOC)
        - Contains TOC indicator keywords
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            return True

        # Check for TOC keywords
        text_lower = text.lower()
        toc_keywords = ['table of contents', 'contents\n', '\ncontents', 'index\n']
        if any(kw in text_lower for kw in toc_keywords):
            return True

        # Count lines that are just numbers (page numbers in TOC)
        number_lines = sum(1 for line in lines if re.match(r'^\d{1,3}$', line))

        # Count very short lines (section numbers like "5.1", single words)
        short_lines = sum(1 for line in lines if len(line) < 10)

        # If more than 40% of lines are standalone numbers → likely TOC
        if len(lines) > 5 and number_lines / len(lines) > 0.4:
            return True

        # If more than 60% of lines are very short → likely TOC or cover
        if len(lines) > 5 and short_lines / len(lines) > 0.6:
            return True

        return False

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting on periods, exclamation, question marks
        # followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s for s in sentences if len(s.strip()) > 20]  # Filter very short sentences
    
    def _is_policy_statement(self, sentence: str) -> bool:
        """
        Check if a sentence is a policy statement
        
        A sentence is considered a policy statement if it:
        - Contains policy-related keywords
        - Has sufficient length
        - Is not just a heading
        """
        sentence_lower = sentence.lower()
        
        # Must be substantial
        if len(sentence) < 30:
            return False
        
        # Check for policy indicators
        policy_indicators = [
            'policy', 'procedure', 'requirement', 'standard', 'guideline',
            'must', 'shall', 'should', 'required', 'mandatory',
            'ensure', 'maintain', 'implement', 'establish', 'define'
        ]
        
        return any(indicator in sentence_lower for indicator in policy_indicators)
    
    def _is_requirement(self, sentence: str) -> bool:
        """Check if sentence contains requirement keywords"""
        return bool(self.requirement_pattern.search(sentence))
    
    def _categorize_statement(self, statement: str) -> Tuple[PolicyCategory, float]:
        """
        Categorize a policy statement
        
        Returns:
            Tuple of (PolicyCategory, confidence_score)
        """
        statement_lower = statement.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in statement_lower)
            if score > 0:
                category_scores[category] = score
        
        if not category_scores:
            return PolicyCategory.GENERAL, 0.5
        
        # Get category with highest score
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]
        
        # Calculate confidence (normalize score)
        confidence = min(0.5 + (max_score * 0.15), 1.0)
        
        return best_category, confidence
    
    def get_statements_by_category(
        self,
        statements: List[PolicyStatement]
    ) -> Dict[PolicyCategory, List[PolicyStatement]]:
        """Group statements by category"""
        categorized = {}
        for statement in statements:
            if statement.category not in categorized:
                categorized[statement.category] = []
            categorized[statement.category].append(statement)
        return categorized
    
    def get_requirements(
        self,
        statements: List[PolicyStatement],
        min_confidence: float = 0.5
    ) -> List[PolicyStatement]:
        """Get only requirement statements with minimum confidence"""
        return [
            s for s in statements
            if s.is_requirement and s.confidence >= min_confidence
        ]
    
    def export_statements(self, statements: List[PolicyStatement]) -> List[Dict]:
        """Export statements to dictionary format"""
        return [
            {
                'text': s.text,
                'category': s.category.value,
                'section': s.section,
                'page': s.page,
                'is_requirement': s.is_requirement,
                'confidence': round(s.confidence, 2)
            }
            for s in statements
        ]
