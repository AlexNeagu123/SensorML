from enum import Enum, auto


class Disease(Enum):
    EARLY_BLIGHT = auto()
    GRAY_MOLD = auto()
    LATE_BLIGHT = auto()
    LEAF_MOLD = auto()
    POWDERY_MILDEW = auto()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


def check_for_air_temperature_disease(air_temperature):
    diseases = []
    if 24 <= air_temperature <= 29:
        diseases.append(Disease.EARLY_BLIGHT)
    if 17 <= air_temperature <= 23:
        diseases.append(Disease.GRAY_MOLD)
    if 10 <= air_temperature <= 24:
        diseases.append(Disease.LATE_BLIGHT)
    if 21 <= air_temperature <= 24:
        diseases.append(Disease.LEAF_MOLD)
    if 22 <= air_temperature <= 30:
        diseases.append(Disease.POWDERY_MILDEW)

    return diseases


def check_for_air_humidity_disease(air_humidity):
    diseases = []
    if 90 <= air_humidity <= 100:
        diseases.extend([Disease.EARLY_BLIGHT, Disease.GRAY_MOLD, Disease.LATE_BLIGHT])
    if 85 <= air_humidity <= 100:
        diseases.append(Disease.LEAF_MOLD)
    if 50 <= air_humidity <= 75:
        diseases.append(Disease.POWDERY_MILDEW)

    return diseases
