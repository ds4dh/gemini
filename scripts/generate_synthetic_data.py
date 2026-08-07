import os
import pandas as pd

SYNTHETIC_DATA = [
    {
        "patient_id": "PAT_001",
        "input_text": (
            "LETTRE DE SORTIE NEUROLOGIE\n"
            "Patient de 64 ans admis pour AIT sylvien gauche. Antécédents d'hypertension artérielle sous Amlodipine 5mg.\n"
            "Habitudes de vie: Consomme environ 15 cigarettes par jour, tabagisme actif non sevré à ce jour.\n"
            "Bilan biologique d'entrée: Créatininémie à 84 µmol/L, glycémie à jeun 5.2 mmol/L, ionogramme sanguin normal. ECG de repos: rythme sinusal à 68/min.\n"
            "Examen neurologique clinique à la sortie: absence totale de déficit résiduel. Le patient a repris son poste d'ingénieur et l'ensemble de ses activités quotidiennes sans aucune gêne ni restriction d'autonomie.\n"
            "Angio-TIRM cérébrale: Découverte d'un anévrisme sacciforme de l'artère communicante antérieure (ACom) mesuré à 0.45 cm de grand axe."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 4.5,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 64,
        "ground_truth_lesion_location": "ACom",
    },
    {
        "patient_id": "PAT_002",
        "input_text": (
            "RAPPORT DE CONSULTATION NEUROCHIRURGIE\n"
            "Mme B., 52 ans, consultée en suivi d'anévrisme cérébral non rompu. Aucune hypertension artérielle dans l'anamnèse (TA 122/78 mmHg).\n"
            "Statut toxique: A définitivement écrasé sa dernière cigarette il y a 6 ans après un sevrage complet.\n"
            "Traitement actuel: Paracétamol 1g si besoin. Biologie de routine sans particularité (plaquettes 240 G/L, hémoglobine 13.5 g/dL).\n"
            "Examen physique: Pas de déficit moteur sévère. Présence d'une incertitude à la marche rapide; la patiente a dû cesser sa pratique sportive intense mais reste parfaitement capable de gérer seule sa toilette, sa cuisine et son logement.\n"
            "TDM cérébral avec injection: Stabilité de l'anévrisme de la bifurcation sylvienne droite (MCA) mesuré à 6.2 mm."
        ),
        "ground_truth_mRS": 2,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 6.2,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 52,
        "ground_truth_lesion_location": "MCA",
    },
    {
        "patient_id": "PAT_003",
        "input_text": (
            "BILAN NEUROLOGIQUE POST-AVC\n"
            "Patient de 78 ans, hémiplégique droit suite à une ischémie cérébrale. Antécédents majeurs d'hypertension artérielle sévère (HTA) traitée par Ramipril et Bisoprolol.\n"
            "Hygiène de vie: Pas d'exposition au tabac, n'a jamais fumé de sa vie.\n"
            "Bilan biologique: Urée 8.1 mmol/L, créatinine 115 µmol/L, cholestérol LDL 1.4 g/L.\n"
            "Évaluation fonctionnelle: Le patient est incapable de marcher sans l'aide physique d'un tiers et nécessite une présence quotidienne constante pour la toilette, l'habillage et le transfert au fauteuil.\n"
            "Bilan d'imagerie vasculaire (ARM): Absence de malformation ou d'anévrisme intracrânien décelable."
        ),
        "ground_truth_mRS": 4,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 78,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_004",
        "input_text": (
            "COMPTE-RENDU D'HOSPITALISATION URGENCE\n"
            "Patient de 45 ans reçu pour céphalées soudaines violentes. Antécédent d'hypertension artérielle connue non suivie régulièrement.\n"
            "Aucune donnée consignée dans l'observation concernant la consommation de tabac.\n"
            "Examens complémentaires: Ponction lombaire positive pour hémorragie sous-arachnoïdienne. NFS: leucocytes 11.2 G/L.\n"
            "Angiographie cérébrale par résonance magnétique: Volumineux anévrisme rompu de l'artère communicante postérieure (PCom) mesuré à 1.1 cm.\n"
            "Statut clinique: Le patient déambule seul sans canne mais présente des troubles cognitifs légers nécessitant l'aide d'un proche pour les démarches administratives et les courses."
        ),
        "ground_truth_mRS": 3,
        "ground_truth_smoking_status": "Unknown",
        "ground_truth_aneurysm_size_mm": 11.0,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 45,
        "ground_truth_lesion_location": "PCom",
    },
    {
        "patient_id": "PAT_005",
        "input_text": (
            "LETTRE DE SUIVI ANÉVRISME\n"
            "Patient de 59 ans, suivi en consultation externe pour anévrisme non rompu. Absence d'antécédent de HTA (tension à 118/74 mmHg).\n"
            "Toxiques: Achète quotidiennement un paquet au bureau de tabac (fumeur actif depuis 30 ans).\n"
            "NFS, créatininémie (78 µmol/L) et bilan d'hémostase normaux.\n"
            "Statut clinique: Discrets engourdissements transitoires du bras gauche sans aucun impact fonctionnel. Autonomie complète préservée au domicile comme au travail.\n"
            "Angio-TC cérébral: Anévrisme du siphon carotidien gauche (ICA) mesuré à 3.8 mm de grand axe."
        ),
        "ground_truth_mRS": 1,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 3.8,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 59,
        "ground_truth_lesion_location": "ICA",
    },
    {
        "patient_id": "PAT_006",
        "input_text": (
            "DISCHARGE SUMMARY - NEUROLOGIE\n"
            "Patiente de 68 ans admise pour paresthésies transitoires du membre supérieur droit. Bilan tensionnel normal sans HTA.\n"
            "Absence totale de facteur de risque toxique inhalé de toute sa vie.\n"
            "Bilan d'exploration: ECG sinusal, troponine négative, glycémie 5.1 mmol/L. Doppler carotidien sans sténose.\n"
            "Symptômes résolus. La patiente réalise l'ensemble de ses activités de ménage et de loisirs sans aucune restriction.\n"
            "Angio-TIRM cérébrale: examen normal, absence d'anévrisme vasculaire."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 68,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_007",
        "input_text": (
            "DISCHARGE SUMMARY - NEUROCHIRURGIE\n"
            "Patient de 71 ans évalué pour anévrisme intracrânien non rompu. Antécédent d'hypertension artérielle sous Valsartan.\n"
            "Mode de vie: A stoppé le tabac lors de son premier infarctus du myocarde en 2015.\n"
            "Biologie: créatinine à 98 µmol/L, bilan hépatique normal.\n"
            "Autonomie: Déambule avec une canne de marche. Ne peut pas porter de charges ni faire le ménage sans l'aide de sa fille.\n"
            "ARM cérébrale: Anévrisme sacciforme de l'apex basilaire (Basilar) mesuré tridimensionnellement à 7.5 x 5.0 mm."
        ),
        "ground_truth_mRS": 3,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 7.5,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 71,
        "ground_truth_lesion_location": "Basilar",
    },
    {
        "patient_id": "PAT_008",
        "input_text": (
            "COMPTE-RENDU NEUROCHIRURGIE URGENCE\n"
            "Patient de 61 ans admis pour hématome intraparenchymateux sylvien. Antécédents de HTA essentielle sévère.\n"
            "Intoxication tabagique importante: fumeur de cigares régulier (25 paquets-années).\n"
            "Scanner cérébral: Hématome cérébral associé à un anévrisme de la bifurcation sylvienne droite (MCA) mesuré à 0.89 cm.\n"
            "État fonctionnel: Patient grabataire, alité en permanence, incontinent et totalement dépendant de l'équipe soignante pour l'hygiène et la nutrition."
        ),
        "ground_truth_mRS": 5,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 8.9,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 61,
        "ground_truth_lesion_location": "MCA",
    },
    {
        "patient_id": "PAT_009",
        "input_text": (
            "CONSULTATION DE NEURO-IMAGERIE\n"
            "Patiente de 38 ans venue pour bilan de céphalées atypiques. Pas de HTA (pression 116/72 mmHg).\n"
            "Patiente non-fumeuse.\n"
            "Bilan sanguin complet normal. Absence de déficit neurologique résiduel ou de limitation d'activité.\n"
            "Angio-TIRM: Petit anévrisme sacculaire de l'artère communicante antérieure (ACom) mesuré à 2.4 mm de diamètre maximal."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": 2.4,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 38,
        "ground_truth_lesion_location": "ACom",
    },
    {
        "patient_id": "PAT_010",
        "input_text": (
            "RAPPORT DE RÉANIMATION NEUROLOGIQUE\n"
            "Patient de 83 ans admis pour coma d'emblée consécutif à une rupture anévrismale majeure. Antécédents de HTA sévère.\n"
            "Ancien fumeur (sevrage tabagique effectué il y a 3 ans).\n"
            "Angioscanner: Hémorragie sous-arachnoïdienne grave avec rupture d'un anévrisme de la carotide interne droite (ICA) mesuré à 1.42 cm.\n"
            "Évolution: Défaillance multi-viscérale et arrêt cardiorespiratoire irréversible. Constat de décès établi à 04h15."
        ),
        "ground_truth_mRS": 6,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 14.2,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 83,
        "ground_truth_lesion_location": "ICA",
    },
    {
        "patient_id": "PAT_011",
        "input_text": (
            "BILAN AMBULATOIRE NEUROLOGIE\n"
            "Patient de 50 ans consulté pour vertiges rotatoires. Pression artérielle non consignée.\n"
            "Renseignements sur le statut tabagique non mentionnés dans le dossier.\n"
            "Examen neurologique normal, autonomie parfaite sans restriction.\n"
            "IRM encéphalique: structures vasculaires sans anomalie, absence d'anévrisme intracrânien."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Unknown",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "Unknown",
        "ground_truth_patient_age": 50,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_012",
        "input_text": (
            "SUIVI NEUROLOGIQUE ANNUEL\n"
            "Patiente de 67 ans suivie pour surveillance vasculo-cérébrale. Antécédent d'hypertension artérielle sous Bisoprolol 5mg.\n"
            "Statut tabagique: Jamais fumé.\n"
            "Discrète fatigabilité lors de la marche prolongée, mais la patiente reste capable d'assumer seule toutes ses activités domestiques.\n"
            "ARM cérébrale: Stabilité d'un anévrisme de l'artère communicante postérieure (PCom) mesuré à 0.51 cm."
        ),
        "ground_truth_mRS": 1,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": 5.1,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 67,
        "ground_truth_lesion_location": "PCom",
    },
    {
        "patient_id": "PAT_013",
        "input_text": (
            "CONSULTATION NEURO-VASCULAIRE\n"
            "Patient de 63 ans vu en bilan de contrôle post-AVC ischémique. Antécédent d'HTA majeure non équilibrée (165/98 mmHg).\n"
            "Consommation tabagique: Fume un demi-paquet de cigarettes par jour depuis plus de 25 ans.\n"
            "Examen neurologique: Présence d'un déficit moteur résiduel du bras droit. Le patient a besoin de l'aide de son épouse pour la toilette matinale et le boutonnage des vêtements.\n"
            "Angio-IRM: Anévrisme fusiforme du tronc basilaire (Basilar) mesuré à 0.95 cm de grand axe."
        ),
        "ground_truth_mRS": 4,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 9.5,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 63,
        "ground_truth_lesion_location": "Basilar",
    },
    {
        "patient_id": "PAT_014",
        "input_text": (
            "NOTE DE SORTIE NEUROCHIRURGIE\n"
            "Patiente de 56 ans hospitalisée pour céphalées chroniques. Antécédent d'hypertension artérielle traitée par Amlodipine.\n"
            "Ancienne consommatrice de tabac ayant définitivement arrêté lors de sa grossesse il y a 12 ans.\n"
            "Statut clinique: Marche autonome sans aide. La patiente gère son ménage mais n'a pas pu reprendre son activité professionnelle en raison d'une fatigabilité cognitive.\n"
            "Angioscanner: Anévrisme du siphon carotidien gauche (ICA) mesuré à 5.0 mm x 4.2 mm."
        ),
        "ground_truth_mRS": 2,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 5.0,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 56,
        "ground_truth_lesion_location": "ICA",
    },
    {
        "patient_id": "PAT_015",
        "input_text": (
            "SUIVI EN MAISON DE RETRAITE / EHPAD\n"
            "Patient de 79 ans évalué pour bilan vasculaire de routine. Absence d'HTA connue (TA 118/72 mmHg).\n"
            "N'a jamais fumé de sa vie (non-fumeur).\n"
            "État fonctionnel: Patient tétraparétique résiduel, constamment alité, porteur d'un étui pénien et totalement dépendant du personnel soignant 24h/24.\n"
            "IRM cérébrale: Découverte d'un anévrisme de l'artère vertébrale gauche (Vertebral) mesuré à 1.15 cm."
        ),
        "ground_truth_mRS": 5,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": 11.5,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 79,
        "ground_truth_lesion_location": "Vertebral",
    },
    {
        "patient_id": "PAT_016",
        "input_text": (
            "COMPTE-RENDU D'HOSPITALISATION AVC\n"
            "Patient de 49 ans admis pour AIT vertébro-basilaire. HTA traitée par Perindopril.\n"
            "Intoxication tabagique active importante évaluée à 20 paquets-années.\n"
            "Examen à la sortie: Récupération ad integrum. Le patient a repris son travail d'artisan et toutes ses activités sportives sans la moindre séquelle fonctionnelle.\n"
            "Angio-TIRM: Absence d'anévrisme vasculaire cérébral."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 49,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_017",
        "input_text": (
            "CONSULTATION NEUROCHIRURGICALE\n"
            "Patient de 74 ans vu pour avis sur image anévrismale. HTA essentielle sous bithérapie.\n"
            "Dossier sans mention quant au statut tabagique.\n"
            "Capacités fonctionnelles: Le patient marche seul avec une canne simple pour sortir mais ne peut pas faire ses courses ou porter ses sacs d'épicerie sans accompagnant.\n"
            "ARM: Anévrisme de l'artère communicante antérieure (ACom) mesuré à 0.68 cm."
        ),
        "ground_truth_mRS": 3,
        "ground_truth_smoking_status": "Unknown",
        "ground_truth_aneurysm_size_mm": 6.8,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 74,
        "ground_truth_lesion_location": "ACom",
    },
    {
        "patient_id": "PAT_018",
        "input_text": (
            "LETTRE DE CONSULTATION EXTERNE\n"
            "Patiente de 41 ans suivie pour dermo-hypodermite et surveillance neurologique. Pas d'HTA.\n"
            "Non-fumeuse (jamais fumé).\n"
            "Symptomatologie: Petite diminution transitoire de la force de préhension de la main droite. Autonomie parfaite préservée pour l'habillage, la cuisine et les soins personnels.\n"
            "Angio-TIRM: Petit anévrisme de la bifurcation sylvienne droite (MCA) mesuré à 3.2 mm."
        ),
        "ground_truth_mRS": 1,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": 3.2,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 41,
        "ground_truth_lesion_location": "MCA",
    },
    {
        "patient_id": "PAT_019",
        "input_text": (
            "COMPTE-RENDU D'HOSPITALISATION - URGENCE\n"
            "Patient de 70 ans admis en urgence pour hémorragie méningée grave. HTA majeure compliquée.\n"
            "Habitudes: Consomme 2 paquets de cigarettes par jour depuis plus de 40 ans.\n"
            "Angioscanner: Rupture d'un anévrisme sylvien gauche (MCA) mesuré à 1.3 cm.\n"
            "Évolution: Enfoncement neurologique irréversible mène au décès du patient au 2ème jour."
        ),
        "ground_truth_mRS": 6,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 13.0,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 70,
        "ground_truth_lesion_location": "MCA",
    },
    {
        "patient_id": "PAT_020",
        "input_text": (
            "BILAN NEURO-VASCULAIRE POST-AIT\n"
            "Patiente de 62 ans examinée en suivi d'ischémie cérébrale. Pas d'HTA (PA 120/78 mmHg).\n"
            "Sevrage tabagique complet et réussi depuis 15 ans.\n"
            "Clinique: Présence d'un discret ralentissement du langage. La patiente est incapable d'exercer son métier de comptable mais gère seule l'ensemble des tâches ménagères et de la vie courante.\n"
            "Angioscanner: Anévrisme du siphon carotidien droit (ICA) mesuré à 0.42 cm."
        ),
        "ground_truth_mRS": 2,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 4.2,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 62,
        "ground_truth_lesion_location": "ICA",
    },
    {
        "patient_id": "PAT_021",
        "input_text": (
            "RAPPORT DE SUIVI ANÉVRISME TRAITÉ\n"
            "Patient de 66 ans revu après embolisation anévrismale. HTA sous Lisinopril.\n"
            "Absence totale de consommation tabagique antérieure.\n"
            "Examen physique: Hémiparésie droite sévère. Le patient est incapable de marcher sans soutien physique et requiert une aide quotidienne importante pour la toilette et l'alimentation.\n"
            "Arthériographie de contrôle: Anévrisme de la communicante postérieure (PCom) mesuré initialement à 8.0 mm, correctement exclu par coils."
        ),
        "ground_truth_mRS": 4,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": 8.0,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 66,
        "ground_truth_lesion_location": "PCom",
    },
    {
        "patient_id": "PAT_022",
        "input_text": (
            "EXAMEN NEUROCHIRURGICAL AMBULATOIRE\n"
            "Patient de 53 ans vu pour surveillance d'image vasculaire. HTA essentielle sous monothérapie.\n"
            "Statut tabagique non renseigné dans la fiche d'admission.\n"
            "Aucun symptôme ni déficit neurologique. Le patient mène une vie active normale sans aucune restriction d'autonomie.\n"
            "Angioscanner: Anévrisme de l'artère vertébrale gauche (Vertebral) mesuré à 0.76 cm."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Unknown",
        "ground_truth_aneurysm_size_mm": 7.6,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 53,
        "ground_truth_lesion_location": "Vertebral",
    },
    {
        "patient_id": "PAT_023",
        "input_text": (
            "CONSULTATION NEUROLOGIQUE\n"
            "Patient de 44 ans évalué pour paraparésie. Pas d'HTA.\n"
            "Tabagisme: Vapoteur quotidien avec reprise récente de 5 à 10 cigarettes par jour.\n"
            "Évaluation fonctionnelle: Patient alité en permanence suite à des séquelles médullaires, sondé à demeure et nécessitant la présence d'une aide-soignante 24h/24 pour tous les soins de base.\n"
            "ARM cérébrale: Absence d'anévrisme vasculaire cérébral décelé."
        ),
        "ground_truth_mRS": 5,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 44,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_024",
        "input_text": (
            "BILAN D'ENTRÉE SOINS INTENSIFS\n"
            "Patiente de 58 ans admise pour céphalées. HTA connue sous trithérapie.\n"
            "Statut toxique: Ancien tabagisme sevré avec succès depuis 2 ans.\n"
            "Examen physique: La patiente déambule seule sans canne et accomplit l'ensemble de ses actes quotidiens, mais décrit de légères céphalées résiduelles intermittentes.\n"
            "Angio-TIRM: Anévrisme de l'artère communicante antérieure (ACom) mesuré à 0.58 cm."
        ),
        "ground_truth_mRS": 1,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 5.8,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 58,
        "ground_truth_lesion_location": "ACom",
    },
    {
        "patient_id": "PAT_025",
        "input_text": (
            "RAPPORT DE CONSULTATION POST-AVC\n"
            "Patient de 72 ans en suivi d'AVC ischémique cérébelleux. Pas d'HTA (TA 120/75 mmHg).\n"
            "Non-fumeur avéré.\n"
            "Clinique: Syndrome cérébelleux persistant. Le patient marche seul avec un déambulateur mais requiert l'aide d'une tierce personne pour effectuer ses achats et entretenir son logement.\n"
            "Angio-TC: Anévrisme de la bifurcation sylvienne droite (MCA) mesuré à 1.05 cm de grand axe."
        ),
        "ground_truth_mRS": 3,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": 10.5,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 72,
        "ground_truth_lesion_location": "MCA",
    },
    {
        "patient_id": "PAT_026",
        "input_text": (
            "RÉANIMATION NEURO-CHIRURGICALE\n"
            "Patient de 65 ans admis pour rupture anévrismale massive. HTA sévère.\n"
            "Consommation toxique: Fumeur actif à raison d'un paquet de cigarettes par jour.\n"
            "Biologie d'urgence: acidose et troubles majeurs de la coagulation.\n"
            "Angiographie: Volumineux anévrisme rompu de la communicante postérieure (PCom) mesuré à 1.6 cm.\n"
            "Évolution: Coma profond d'emblée suivi d'un arrêt cardiaque réfractaire. Décès constaté."
        ),
        "ground_truth_mRS": 6,
        "ground_truth_smoking_status": "Smoker",
        "ground_truth_aneurysm_size_mm": 16.0,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 65,
        "ground_truth_lesion_location": "PCom",
    },
    {
        "patient_id": "PAT_027",
        "input_text": (
            "RAPPORT D'EXPLORATION VASCULAIRE\n"
            "Patient de 60 ans vu pour bilan de suivi d'anévrisme du tronc basilaire. HTA sous IEC.\n"
            "Mode de vie: Ancien fumeur ayant arrêté la cigarette il y a plus de 12 ans.\n"
            "Examen neurologique: Présence d'une légère boiterie du côté gauche. Le patient conserve néanmoins une indépendance fonctionnelle absolue pour l'ensemble des activités domestiques et professionnelles.\n"
            "ARM: Stabilité de l'anévrisme du tronc basilaire (Basilar) mesuré à 4.8 mm."
        ),
        "ground_truth_mRS": 2,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": 4.8,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 60,
        "ground_truth_lesion_location": "Basilar",
    },
    {
        "patient_id": "PAT_028",
        "input_text": (
            "LETTRE DE CONSULTATION NEUROLOGIQUE\n"
            "Patiente de 35 ans explorée pour céphalées cataméniales. Bilan tensionnel normal.\n"
            "Pas d'histoire de tabagisme (jamais fumé).\n"
            "Examen clinique parfaitement normal, absence de restriction fonctionnelle ou de symptôme neurologique.\n"
            "Angio-TIRM cérébrale: Réseau vasculaire intracrânien harmonieux, absence d'anévrisme décelé."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "No",
        "ground_truth_patient_age": 35,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_029",
        "input_text": (
            "LETTRE DE SUIVI EN CONSULTATION EXTERNE\n"
            "Patient de 69 ans vu pour contrôle annuel de son hypertension artérielle. TA mesurée à 132/80 mmHg sous Lercanidipine.\n"
            "Statut toxique: Ancien fumeur, sevrage tabagique complet depuis 8 ans.\n"
            "Examen neurologique: Absence de symptôme neurologique résiduel. Le patient marche seul et réalise l'ensemble des tâches quotidiennes en toute autonomie.\n"
            "Examen complémentaire: Bilan biologique sanguin et ECG normaux. Aucune imagerie cérébrale ni angioscanner vasculaire n'a été réalisé lors de cette consultation de routine."
        ),
        "ground_truth_mRS": 0,
        "ground_truth_smoking_status": "Former-smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 69,
        "ground_truth_lesion_location": "None",
    },
    {
        "patient_id": "PAT_030",
        "input_text": (
            "RAPPORT DE CONSULTATION DE SUIVI CLINIQUE\n"
            "Patiente de 54 ans consultée pour suivi de tension artérielle. HTA traitée par Irbésartan 150mg.\n"
            "Habitudes: N'a jamais fumé de sa vie (non-fumeuse).\n"
            "Évaluation fonctionnelle: La patiente décrit une fatigabilité précoce au travail mais reste autonome pour la toilette, la cuisine et les déplacements au quotidien.\n"
            "Aucun bilan d'imagerie cérébrale ni mesure anévrismale n'a été effectué au cours de cette visite."
        ),
        "ground_truth_mRS": 1,
        "ground_truth_smoking_status": "Non-smoker",
        "ground_truth_aneurysm_size_mm": None,
        "ground_truth_hypertension": "Yes",
        "ground_truth_patient_age": 54,
        "ground_truth_lesion_location": "None",
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
