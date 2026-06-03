from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path("data/synthetic_corpus.jsonl")


def build_doc(doc_id: str, doc_type: str, title: str, jurisdiction: str, s1: str, s2: str) -> dict:
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": title,
        "jurisdiction": jurisdiction,
        "sections": [
            {"section_id": "S1", "title": "Scope", "text": s1},
            {"section_id": "S2", "title": "Control", "text": s2},
        ],
    }


def main() -> None:
    docs = [
        build_doc("POL-001", "policy", "KYC Minimum Controls", "global", "Customer onboarding requires identity document verification and sanctions list screening before account activation.", "Enhanced due diligence is mandatory for politically exposed persons and high-risk jurisdictions."),
        build_doc("POL-002", "policy", "Transaction Monitoring", "global", "Alerts must be reviewed within one business day by level-1 analysts.", "Escalated cases require level-2 signoff and SAR decision within five business days."),
        build_doc("POL-003", "policy", "Record Retention", "global", "Compliance records must be retained for seven years.", "Deletion requests are blocked for records under legal hold or active audit."),
        build_doc("POL-004", "policy", "Model Governance", "global", "AI outputs for compliance are recommendations and require human approval.", "Any override must be documented with rationale and reviewer identity."),
        build_doc("POL-005", "policy", "Complaints Handling", "US", "Customer complaints involving potential UDAAP risk are triaged in 24 hours.", "Root cause analysis must be completed in 10 business days."),
        build_doc("POL-006", "policy", "Third-Party Risk", "global", "Critical vendors require annual control attestation.", "Material findings must have remediation plans approved by risk committee."),
        build_doc("POL-007", "policy", "Data Residency", "EU", "EU customer identifiers must remain in EU-hosted systems unless legal basis is approved.", "Cross-border transfer requires DPO review and standard contractual clauses."),
        build_doc("POL-008", "policy", "Insider Trading Controls", "US", "Watch list updates occur daily before market open.", "Employee trades in restricted securities require pre-clearance."),
        build_doc("AUD-001", "audit", "Q1 AML Audit Findings", "global", "Sample testing found 6 of 100 alerts exceeded SLA by more than 48 hours.", "Remediation required workflow automation and queue balancing."),
        build_doc("AUD-002", "audit", "KYC File Completeness", "global", "12 percent of files lacked secondary address verification evidence.", "Issue owner must close gaps before next quarterly review."),
        build_doc("AUD-003", "audit", "SAR Decision Timeliness", "US", "Late SAR decisions were linked to unclear escalation ownership.", "Control enhancement added explicit level-2 duty rota."),
        build_doc("AUD-004", "audit", "Data Lineage Audit", "EU", "Lineage for sanctions screening inputs was incomplete in two pipelines.", "Engineering committed metadata catalog fixes within 30 days."),
        build_doc("AUD-005", "audit", "Access Control Audit", "global", "Three privileged accounts lacked quarterly recertification evidence.", "Temporary exception approved for seven days only."),
        build_doc("AUD-006", "audit", "Model Explainability Audit", "global", "Analyst notes were missing for 9 percent of AI-assisted cases.", "Checklist updated to require evidence citation in every decision."),
        build_doc("REG-001", "regulation", "AML Bulletin 24-01", "global", "Institutions must apply risk-based customer due diligence.", "Monitoring systems should support timely suspicious activity escalation."),
        build_doc("REG-002", "regulation", "Consumer Duty Update", "UK", "Firms must demonstrate good outcomes for retail customers.", "Boards should receive regular outcome metrics and challenge records."),
        build_doc("REG-003", "regulation", "EU Transfer Guidance", "EU", "Cross-border personal data transfer requires legal transfer mechanism and risk assessment.", "Supervisory authorities may request transfer impact assessments."),
        build_doc("REG-004", "regulation", "US CFPB Circular", "US", "Unfair, deceptive, or abusive acts must be prevented through proactive controls.", "Complaint trend analysis is expected as an early warning mechanism."),
        build_doc("REG-005", "regulation", "Operational Resilience Bulletin", "global", "Critical compliance services require tested continuity plans.", "Material disruptions must be reported to regulators without undue delay."),
        build_doc("REG-006", "regulation", "AI Governance Note", "global", "Automated decision support in regulated functions must remain under accountable human oversight.", "Firms should evidence traceability from recommendation to final decision."),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(docs)} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
