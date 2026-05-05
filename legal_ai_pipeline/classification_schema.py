"""Legal issue label schema for Pipeline 2."""

from __future__ import annotations

from .classification_config import ClassificationConfig
from .io_utils import ensure_dir, write_json


LABEL_SCHEMA = {
    "constitutional_law": {
        "name": "Constitutional Law",
        "description": "Fundamental rights, constitutional remedies, articles, and constitutional powers.",
        "keywords": ["constitution", "constitutional", "article", "fundamental", "rights", "article 14", "article 21"],
    },
    "criminal_law": {
        "name": "Criminal Law",
        "description": "Offences, conviction, sentence, bail, police investigation, and criminal procedure.",
        "keywords": ["criminal", "accused", "conviction", "sentence", "offence", "murder", "bail", "police", "ipc"],
    },
    "civil_procedure": {
        "name": "Civil Procedure",
        "description": "Civil suits, decrees, appeals, injunctions, limitation, and procedural orders.",
        "keywords": ["civil", "suit", "decree", "injunction", "appeal", "plaintiff", "defendant", "limitation"],
    },
    "property_law": {
        "name": "Property Law",
        "description": "Land, possession, tenancy, transfer, ownership, acquisition, and property disputes.",
        "keywords": ["property", "land", "possession", "tenant", "lease", "acquisition", "ownership", "transfer"],
    },
    "contract_law": {
        "name": "Contract Law",
        "description": "Agreements, contractual obligations, consideration, breach, damages, and commercial claims.",
        "keywords": ["contract", "agreement", "breach", "consideration", "damages", "commercial", "arbitration"],
    },
    "service_law": {
        "name": "Service Law",
        "description": "Employment, public service, recruitment, promotion, dismissal, pension, and service benefits.",
        "keywords": ["service", "employment", "employee", "promotion", "dismissal", "pension", "appointment", "recruitment"],
    },
    "administrative_law": {
        "name": "Administrative Law",
        "description": "Government action, public authorities, tribunals, delegated powers, and administrative review.",
        "keywords": ["government", "authority", "administrative", "tribunal", "public", "notification", "commission"],
    },
    "evidence_law": {
        "name": "Evidence Law",
        "description": "Witnesses, admissibility, proof, documents, burden, and appreciation of evidence.",
        "keywords": ["evidence", "witness", "proof", "documentary", "admissible", "testimony", "burden"],
    },
    "family_law": {
        "name": "Family Law",
        "description": "Marriage, divorce, maintenance, succession, adoption, guardianship, and family rights.",
        "keywords": ["marriage", "divorce", "maintenance", "succession", "adoption", "guardian", "family", "wife"],
    },
    "tax_law": {
        "name": "Tax Law",
        "description": "Income tax, sales tax, excise, customs, assessment, and revenue disputes.",
        "keywords": ["tax", "income tax", "sales tax", "excise", "customs", "assessment", "revenue"],
    },
    "labor_law": {
        "name": "Labor Law",
        "description": "Industrial disputes, wages, workmen, factories, labour welfare, and employment conditions.",
        "keywords": ["labour", "labor", "workman", "industrial", "wages", "factory", "union"],
    },
    "writ_jurisdiction": {
        "name": "Writ Jurisdiction",
        "description": "Writ petitions and constitutional remedies such as mandamus, certiorari, habeas corpus, and prohibition.",
        "keywords": ["writ", "mandamus", "certiorari", "habeas", "corpus", "prohibition", "article 226", "article 32"],
    },
    "statutory_interpretation": {
        "name": "Statutory Interpretation",
        "description": "Meaning, construction, applicability, sections, articles, rules, regulations, and statutory powers.",
        "keywords": ["section", "act", "rule", "rules", "regulation", "interpretation", "statute", "provision"],
    },
    "other": {
        "name": "Other",
        "description": "Fallback label when the issue is not clearly covered by the main taxonomy.",
        "keywords": [],
    },
}


def create_label_schema(config: ClassificationConfig) -> dict:
    ensure_dir(config.labels_dir)
    schema = {
        "multi_label": True,
        "labels": LABEL_SCHEMA,
        "defaults": {
            "confidence_threshold": config.confidence_threshold,
            "max_labels_per_item": config.max_labels_per_item,
        },
    }
    write_json(config.labels_dir / "label_schema.json", schema)
    return schema
