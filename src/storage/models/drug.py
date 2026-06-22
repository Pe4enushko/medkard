from dataclasses import dataclass


@dataclass
class Drug:
    trade_name: str
    inn_name: str | None = None
    dosage_form: str | None = None
    dosage: str | None = None
    patient_exclusions: str | None = None

    id: int | None = None
