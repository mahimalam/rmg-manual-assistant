"""Shared extraction records. Shape validity is separate from factual review."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "knowledge-v1"


class UnitDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["information", "procedure", "troubleshooting", "table", "figure"]
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=16000)
    source_quote: str = ""
    bbox: tuple[float, float, float, float] | None = None
    condition: str = ""
    cause: str | None = None
    checks: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[list[str | None]] = Field(default_factory=list)
    cell_boxes: list[list[tuple[float, float, float, float] | None]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    references: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    parent_key: str | None = None
    table_key: str | None = None
    section_key: str = ""

    @model_validator(mode="after")
    def relationships(self):
        if self.kind == "table":
            if not self.table_headers or not self.table_rows:
                raise ValueError("A table needs headers and rows")
            if any(len(row) != len(self.table_headers) for row in self.table_rows):
                raise ValueError("Table cells do not match column headers")
            if self.cell_boxes and (len(self.cell_boxes) != len(self.table_rows) or
                                    any(len(row) != len(self.table_headers) for row in self.cell_boxes)):
                raise ValueError("Cell locations do not match the table shape")
        if self.kind == "troubleshooting" and not (self.checks or self.actions):
            raise ValueError("A troubleshooting branch needs checks or actions")
        if self.bbox and (self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]):
            raise ValueError("Invalid source rectangle")
        return self

    def build_search_text(self):
        parts = [self.title, self.text]
        if self.condition:
            parts.append(f"Condition: {self.condition}")
        if self.cause:
            parts.append(f"Cause: {self.cause}")
        for label, values in (("Check", self.checks), ("Action", self.actions),
                              ("Prerequisite", self.prerequisites), ("Warning", self.warnings)):
            parts.extend(f"{label}: {value}" for value in values)
        if self.labels:
            parts.append("Labels: " + "; ".join(f"{key}: {value}" for key, value in self.labels.items()))
        parts.extend(f"Footnote: {value}" for value in self.footnotes)
        return "\n".join(dict.fromkeys(parts))


class PageDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: list[UnitDraft] = Field(max_length=150)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manual_id: str = Field(min_length=1)
    pdf_page_index: int = Field(ge=0)
    printed_page_label: str
    bbox: tuple[float, float, float, float] | None


class Dependency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    relation: Literal["warning", "prerequisite", "procedure", "table", "figure", "alternative", "note", "applicability"]
    required_value: str | None = None
    when_value: str | None = None


class SettingRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    setting_id: str
    value: str


class SettingExclusion(SettingRequirement):
    when_value: str


class SettingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    source_quote: str
    effect: Literal['neutral_drop','treadle_operation'] | None
    enabled: bool | None
    trigger: Literal['after_knee_switch','neutral_after_thread_trim'] | None
    description: str
    conditions: list[SettingRequirement] = Field(default_factory=list)
    exclusions: list[SettingRequirement] = Field(default_factory=list)


class KnowledgeUnit(UnitDraft):
    id: str = Field(min_length=1)
    manual_id: str = Field(min_length=1)
    manual_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page: int = Field(ge=1)
    printed_page_label: str
    parent_id: str | None
    source_refs: list[SourceRef] = Field(min_length=1)
    search_text: str = Field(min_length=1)
    extraction_method: Literal["text", "ocr-eng", "vlm"]
    review_status: Literal["unreviewed", "user_checked", "domain_approved", "rejected"]
    flags: list[str]
    blocking_issues: list[str]
    dependencies: list[Dependency]
    applicable_profile_ids: list[str] = Field(default_factory=list)
    allowed_role: Literal["unknown", "operator", "technician"] = "unknown"
    note_ids: list[str] = Field(default_factory=list)
    note_references: list[str] = Field(default_factory=list)
    setting_id: str | None = None
    setting_value: str | None = None
    setting_requirements: list[SettingRequirement] = Field(default_factory=list)
    setting_exclusions: list[SettingExclusion] = Field(default_factory=list)
    setting_rules: list[SettingRule] = Field(default_factory=list)
    applicable_models: list[str] = Field(default_factory=list)
    excluded_models: list[str] = Field(default_factory=list)
    is_applicability_rule: bool = False

    @model_validator(mode="after")
    def source_identity(self):
        if (any(source.manual_id != self.manual_id for source in self.source_refs) or
                not any(source.pdf_page_index == self.page - 1 for source in self.source_refs)):
            raise ValueError("Source references do not match the knowledge unit")
        if any(dependency.id == self.id for dependency in self.dependencies):
            raise ValueError("A unit cannot require itself")
        return self
