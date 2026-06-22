from dataclasses import dataclass


@dataclass
class DietarySupplement:
    product_name: str
    registration_number: str | None = None
    status: str | None = None
    manufacturer_name: str | None = None
    country_of_manufacture: str | None = None
    scope_of_application: str | None = None
    label_info: str | None = None
    registered_at: str | None = None  # ISO date string, e.g. "2024-01-15"

    id: int | None = None
