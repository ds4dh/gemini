import os
import pandas as pd

SYNTHETIC_DATA = [
    {
        "patient_id": "PAT_001",
        "input_text": (
            "LETTRE DE SORTIE NEUROLOGIE\n"
            "Patient de 64 ans admis pour AIT sylvien gauche. Examen clinique sans déficit résiduel à la sortie. "
            "Le patient a repris l'ensemble de ses activités quotidiennes sans aucune gêne (mRS 0). "
            "Habitudes: Tabagisme actif à 20 paquets-années, non sevré à ce jour. "
            "Angio-TIRM cérébrale: Découverte d'un anévrisme sacciforme de l'artère communicante antérieure mesuré à 4.5 mm de grand axe."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 4.5,
    },
    {
        "patient_id": "PAT_002",
        "input_text": (
            "RAPPORT DE CONSULTATION NEUROCHIRURGIE\n"
            "Mme B., 52 ans, consultée en suivi d'anévrisme cérébral non rompu. "
            "Pas de déficit moteur ni sensitif. Incapacité légère: la patiente a dû cesser sa pratique sportive intense mais reste autonome pour les actes de la vie quotidienne. "
            "Statut tabagique: Ancien fumeur, a arrêté le tabac il y a 5 ans. "
            "TDM cérébral de contrôle: Stabilité de l'anévrisme de la bifurcation sylvienne droite mesuré à 6.2 mm."
        ),
        "ground_truth_mRS": 2,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 6.2,
    },
    {
        "patient_id": "PAT_003",
        "input_text": (
            "BILAN NEUROLOGIQUE POST-AVC\n"
            "Patient de 78 ans, hémiplégique droit suite à une ischémie cérébrale. "
            "Incapacité modérément sévère: le patient est incapable de marcher sans assistance et nécessite une aide quotidienne pour ses soins corporels (mRS 4). "
            "Le patient n'a jamais fumé de sa vie (non-fumeur). "
            "Bilan d'imagerie vasculaire: Absence d'anévrisme intracrânien décelable."
        ),
        "ground_truth_mRS": 4,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": None,
    },
    {
        "patient_id": "PAT_004",
        "input_text": (
            "COMPTE-RENDU D'HOSPITALISATION\n"
            "Patient de 45 ans reçu pour céphalées soudaines en coup de tonnerre. "
            "Acanthocytose et ponction lombaire positives pour hémorragie sous-arachnoïdienne. "
            "Angiographie cérébrale: Volumineux anévrisme rompu de la PICA mesuré à 11.0 mm. "
            "Aucune information mentionnée dans le dossier concernant la consommation de tabac."
        ),
        "ground_truth_mRS": 3,
        "ground_truth_smoking_status": "Unknown",
        "ground_truth_aneurysm_size_mm": 11.0,
    },
    {
        "patient_id": "PAT_005",
        "input_text": (
            "LETTE DE SUIVI ANÉVRISME\n"
            "Patient de 59 ans, suivi pour anévrisme incultable. "
            "Actuellement asymptomatique, autonomie complète au domicile et au travail (mRS 1). "
            "Le patient fume 15 cigarettes par jour (fumeur actif). "
            "Angio-TC: Anévrisme du siphon carotidien gauche de 3.8 mm."
        ),
        "ground_truth_mRS": 1,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 3.8,
    },
    {
        "patient_id": "PAT_006",
        "input_text": (
            "DISCHARGE SUMMARY - NEUROLOGY\n"
            "68-year-old female presenting with mild right arm weakness after TIA. "
            "Symptoms completely resolved. Patient performs all usual duties without restriction (mRS 0). "
            "Never smoker. CTA shows no intracranial aneurysm."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": None,
    },
    {
        "patient_id": "PAT_007",
        "input_text": (
            "DISCHARGE SUMMARY - NEUROSURGERY\n"
            "71-year-old male evaluated for unruptured intracranial aneurysm. "
            "Former smoker (stopped 10 years ago). Moderate disability (mRS 3): walks with cane, requires assistance for shopping. "
            "MRA demonstrates a 7.5 mm basilar tip aneurysm."
        ),
        "ground_truth_mRS": 3,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 7.5,
    },
]


def generate_synthetic_dataset(output_path: str = "data/synthetic_clinical_notes.csv") -> str:
    """
    Generates a synthetic clinical dataset CSV for pipeline tests and benchmarks.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(SYNTHETIC_DATA)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated synthetic dataset with {len(df)} samples at '{output_path}'")
    return output_path


if __name__ == "__main__":
    generate_synthetic_dataset()
