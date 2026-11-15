"""Spray prescription engine (fixed version).

IMPORTANT: the chemical names, doses and prices below are PLACEHOLDER values.
Always verify them against the product label and a local agronomist before
anyone sprays a real field.
"""
import math

DISCLAIMER = (
    "Doses and prices are example values only. Check the product label or ask "
    "a local agriculture officer before spraying."
)


CLASS_TO_DISEASE = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy",
}

AGRONOMIC_DATABASE = {
    "Early Blight": {
        "chemical_name": "Mancozeb 75% WP",
        "dosage_per_acre_grams": 500.0,
        "water_per_acre_liters": 100.0,
        "cost_per_kg_pkr": 1800.0,
    },
    "Late Blight": {
        "chemical_name": "Metalaxyl + Mancozeb",
        "dosage_per_acre_grams": 600.0,
        "water_per_acre_liters": 120.0,
        "cost_per_kg_pkr": 2200.0,
    },
    "Healthy": {
        "chemical_name": "None",
        "dosage_per_acre_grams": 0.0,
        "water_per_acre_liters": 0.0,
        "cost_per_kg_pkr": 0.0,
    },
}

HECTARE_TO_ACRE = 2.47105


def class_to_disease(class_name: str) -> str:
    """Convert a model class name to a database key."""
    if class_name not in CLASS_TO_DISEASE:
        raise ValueError(f"No database entry for class '{class_name}'")
    return CLASS_TO_DISEASE[class_name]


def calculate_prescription(disease_name, field_area, unit="acre", tank_capacity=20.0):
    """Return a full spray report. Always returns the SAME keys.

    Raises ValueError for bad input (unknown disease, area <= 0, tank <= 0).
    """
    if disease_name not in AGRONOMIC_DATABASE:
        raise ValueError(f"Unknown disease: {disease_name}")
    if field_area is None or field_area <= 0:
        raise ValueError("Field area must be greater than 0")
    if tank_capacity is None or tank_capacity <= 0:
        raise ValueError("Tank capacity must be greater than 0")

    unit_clean = unit.strip().lower()
    acres = field_area * HECTARE_TO_ACRE if unit_clean in ("hectare", "hectares", "ha") else field_area

    info = AGRONOMIC_DATABASE[disease_name]

    if disease_name == "Healthy":
        return {
            "disease": disease_name,
            "needs_treatment": False,
            "message": "Crop looks healthy. No chemical treatment required.",
            "acres": round(acres, 2),
            "chemical": "None",
            "water_liters": 0.0,
            "chemical_grams": 0.0,
            "total_tanks": 0,
            "grams_per_tank": 0.0,
            "cost_pkr": 0.0,
        }

    total_water = info["water_per_acre_liters"] * acres
    total_chem_g = info["dosage_per_acre_grams"] * acres
    total_cost = (total_chem_g / 1000.0) * info["cost_per_kg_pkr"]

    total_tanks = math.ceil(total_water / tank_capacity)
    chem_per_tank = total_chem_g / total_tanks

    return {
        "disease": disease_name,
        "needs_treatment": True,
        "message": f"Treatment recommended for {disease_name}.",
        "acres": round(acres, 2),
        "chemical": info["chemical_name"],
        "water_liters": round(total_water, 2),
        "chemical_grams": round(total_chem_g, 2),
        "total_tanks": total_tanks,
        "grams_per_tank": round(chem_per_tank, 2),
        "cost_pkr": round(total_cost, 2),
    }
